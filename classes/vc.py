# Import your module
import os
import random
import sys
import lib
import time
import torch
import soundfile as sf
import asyncio
import traceback
import torchaudio
import bittensor as bt
import datetime as dt
from tabulate import tabulate
from datasets import load_dataset
import lib.protocol
from lib.protocol import VoiceClone
from lib.clone_score import CloneScore
from classes.aimodel import AIModelService
import wandb
import numpy as np

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)
# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)


class VoiceCloningService(AIModelService):
    # Existing __init__ and other methods...
    def __init__(self):
        super().__init__()  # Initializes base class components
        self.load_vc_prompts()
        self.load_vc_voices()
        self.total_dendrites_per_query = self.vcdnp  # Example value, adjust as needed
        self.minimum_dendrites_per_query = 5  # Example value, adjust as needed
        self.combinations = []
        self.lock = asyncio.Lock()

        ###################################### DIRECTORY STRUCTURE ###########################################
        self.source_path = os.path.join(audio_subnet_path, "vc_source")
        # Check if the directory exists
        if not os.path.exists(self.source_path):
            # If not, create the directory
            os.makedirs(self.source_path)
        self.target_path = os.path.join(audio_subnet_path, "vc_target")
        # Check if the directory exists
        if not os.path.exists(self.target_path):
            # If not, create the directory
            os.makedirs(self.target_path)
        self.processed_path = os.path.join(audio_subnet_path, "vc_processed")
        # Check if the directory exists
        if not os.path.exists(self.processed_path):
            # If not, create the directory
            os.makedirs(self.processed_path)
        ###################################### DIRECTORY STRUCTURE ###########################################
        self.filtered_axon = []
        self.filtered_axons = []
        self.response = None
        self.filename = ""
        self.audio_file_path = ""
        self.text_input = ""

    def load_vc_prompts(self):
        gs_dev = load_dataset("etechgrid/Prompts_for_Voice_cloning_and_TTS")
        self.prompts = gs_dev['train']['text']
        return self.prompts

    def load_vc_voices(self):
        dataset = load_dataset("etechgrid/28.5k_wavfiles_dataset")
        if 'train' in dataset:
            self.audio_files = [item['audio'] for item in dataset['train']]
            return self.audio_files

    async def run_async(self):
        step = 0
        running_tasks = []
        while True:
            try:
                new_tasks = await self.main_loop_logic(step)
                running_tasks.extend(new_tasks)

                # Periodically check and clean up completed tasks
                running_tasks = [task for task in running_tasks if not task.done()]
                step += 1

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting VoiceCloneService.")
                break
            except Exception as e:
                print(f"An error occurred in VoiceCloneService: {e}")
                traceback.print_exc()
    async def process_huggingface_prompts(self, step):
        if step % 45 == 0:
            async with self.lock:
                bt.logging.info(f"--------------------------------- Prompt and voices are being used from HuggingFace Dataset for Voice Clone at Step: {step} ---------------------------------")
                self.filename = ""
                self.text_input = random.choice(self.prompts)
                while len(self.text_input) > 256:
                    bt.logging.error(f"The length of current Prompt is greater than 256. Skipping current prompt.")
                    self.text_input = random.choice(self.prompts)

                vc_voice = random.choice(self.audio_files)
                audio_array = vc_voice['array']
                sampling_rate = vc_voice['sampling_rate']
                self.hf_voice_id = vc_voice['path'].split("/")[-1][:10]
                sf.write('input_file.wav', audio_array, sampling_rate)
                self.audio_file_path = os.path.join(audio_subnet_path, "input_file.wav")
                waveform, _ = torchaudio.load(self.audio_file_path)
                clone_input = waveform.tolist()
                sample_rate = sampling_rate
                await self.generate_voice_clone(self.text_input, clone_input, sample_rate)

    async def process_local_files(self, step, sound_files):
        if step % 25 == 0 and sound_files:
            bt.logging.info(f"--------------------------------- Prompt and voices are being used locally for Voice Clone at Step: {step} ---------------------------------")
            # Extract the base name (without extension) of each sound file
            sound_file_basenames = [os.path.splitext(f)[0] for f in sound_files]
            for filename in sound_files:
                self.filename = filename
                text_file = os.path.splitext(filename)[0] + ".txt"
                text_file_path = os.path.join(self.source_path, text_file)
                self.audio_file_path = os.path.join(self.source_path, filename)
                new_file_path = os.path.join(self.processed_path, filename)
                new_txt_path = os.path.join(self.processed_path, text_file)

                
                # Check if the base name of the text file is in the list of sound file base names
                if os.path.splitext(text_file)[0] in sound_file_basenames:
                    with open(text_file_path, 'r') as file:
                        text_content = file.read().strip()
                        self.text_input = text_content
                    if len(self.text_input) > 256:
                        bt.logging.error(f"The length of current Prompt is greater than 256. Skipping current prompt.")
                        continue
                    audio_content, sampling_rate = self.read_audio_file(self.audio_file_path)
                    clone_input = audio_content.tolist()
                    sample_rate = sampling_rate
                    self.hf_voice_id = "local" 
                    await self.generate_voice_clone(self.text_input,clone_input, sample_rate)

                    # Move the file to the processed directory
                    if os.path.exists(self.audio_file_path):
                        os.rename(self.audio_file_path, new_file_path)
                        os.rename(text_file_path, new_txt_path)
                    else:
                        bt.logging.warning(f"File not found: {self.audio_file_path}, it may have already been processed.")
                    # Move the text file to the processed directory
            
            bt.logging.info("All files have been successfully processed from the vc_source directory.")
            


    async def main_loop_logic(self, step):
        tasks = []
        try:
            files = os.listdir(self.source_path)
            sound_files = [f for f in files if f.endswith(".wav") or f.endswith(".mp3")]

            # Schedule both tasks to run concurrently
            huggingface_task = asyncio.create_task(self.process_huggingface_prompts(step))
            local_files_task = asyncio.create_task(self.process_local_files(step, sound_files))
            tasks.extend([huggingface_task, local_files_task])

        except Exception as e:
            bt.logging.error(f"An error occurred in VoiceCloningService: {e}")
            traceback.print_exc()

        await asyncio.sleep(0.5)  # Delay at the end of each loop iteration
        return tasks

    def convert_array_to_wav(audio_data, output_filename):
        """
        Converts an audio data array to a .wav file.

        Parameters:
        audio_data (dict): A dictionary containing 'array' and 'sampling_rate'.
        output_filename (str): The desired output filename for the .wav file.

        Returns:
        str: The path to the generated .wav file.
        """
        try:
            # Extract array and sampling_rate from audio_data
            audio_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']

            # Write the data to a .wav file
            sf.write(output_filename, audio_array, sampling_rate)
            print(f"Successfully saved the waveform to {output_filename}")
            return output_filename
        except KeyError as e:
            print(f"KeyError: Make sure that 'array' and 'sampling_rate' are in audio_data. Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def read_audio_file(self, path):
        try:
            # Read the audio file and return its content
            waveform, sample_rate = torchaudio.load(path)
            return waveform, sample_rate
        except Exception as e:
            print(f"An error occurred while reading the audio file: {e}")
    

    async def generate_voice_clone(self, text_input, clone_input, sample_rate):
        try:
            self.filtered_axons = self.get_filtered_axons_from_combinations()
            for ax in self.filtered_axons:
                self.response = await self.dendrite.forward(
                    ax,
                    lib.protocol.VoiceClone(text_input=text_input, clone_input=clone_input, sample_rate=sample_rate,hf_voice_id=self.hf_voice_id),
                    deserialize=True,
                    timeout=130
                )
                # Process the responses if needed
                self.process_voice_clone_responses(ax)
            bt.logging.info(f"Updated Scores for Voice Cloning: {self.scores}")
        except Exception as e:
            print(f"An error occurred while processing the voice clone: {e}")

    def process_voice_clone_responses(self, ax):
        try:
            if self.response is not None and isinstance(self.response, lib.protocol.VoiceClone) and self.response.clone_output is not None and self.response.dendrite.status_code == 200:
                bt.logging.success(f"Received Voice Clone output from {ax.hotkey}")
                self.handle_clone_output(ax, self.response)
            elif self.response.dendrite.status_code != 403:
                self.punish(ax, service="Voice Cloning", punish_message=self.response.dendrite.status_message)
            else:
                pass
            return ax.hotkey
        except Exception as e:
            print(f"An error occurred while processing voice clone responses: {e}")

    def handle_clone_output(self, axon, response):
        try:
            if response is not None and response.clone_output is not None:
                output = response.clone_output
                # Convert the list to a tensor
                clone_tensor = torch.Tensor(output)

                # Normalize the speech data
                audio_data = clone_tensor / torch.max(torch.abs(clone_tensor))
                # Convert to 32-bit PCM
                audio_data_int_ = (audio_data * 2147483647).type(torch.IntTensor)
                # Add an extra dimension to make it a 2D tensor
                audio_data_int = audio_data_int_.unsqueeze(0)
                if response.model_name == "elevenlabs/eleven":
                    sampling_rate = 44000
                else:
                    sampling_rate = 24000
                file = self.filename.split(".")[0]
                cloned_file_path = os.path.join(self.target_path, file + '_cloned_'+ axon.hotkey[:6] +'_.wav' )
                if file is None or file == "":
                    cloned_file_path = os.path.join('/tmp', self.hf_voice_id + '_cloned_'+ axon.hotkey[:6] +'_.wav' )
                torchaudio.save(cloned_file_path, src=audio_data_int, sample_rate=sampling_rate)
                try:
                    uid_in_metagraph = self.metagraph.hotkeys.index(axon.hotkey)
                    wandb.log({f"Voice Clone Prompt: {response.text_input}": wandb.Audio(np.array(audio_data_int_), caption=f'For HotKey: {axon.hotkey[:10]} and uid {uid_in_metagraph}', sample_rate=sampling_rate)})
                    bt.logging.success(f"Voice Clone Audio file uploaded to wandb successfully for Hotkey {axon.hotkey} and uid {uid_in_metagraph}")
                except Exception as e:
                    bt.logging.error(f"Error uploading Voice Clone Audio file to wandb: {e}")                               
                # Score the output and update the weights
                score = self.score_output(self.audio_file_path, cloned_file_path, self.text_input)
                self.update_score(axon, score, service="Voice Cloning", ax=self.filtered_axon)
                existing_wav_files = [f for f in os.listdir('/tmp') if f.endswith('.wav')]
                for existing_file in existing_wav_files:
                    try:
                        os.remove(os.path.join('/tmp', existing_file))
                    except Exception as e:
                        bt.logging.error(f"Error deleting existing WAV file: {e}")

        except Exception as e:
            pass
            # bt.logging.info(f"Error processing speech output : {e}")


    def score_output(self, input_path, output_path, text_input):
        '''Score the output based on the input and output paths'''
        try:
            clone_score = CloneScore()
            # Call the scoring function from lib.reward
            score, max_mse = clone_score.compare_audio(input_path , output_path, text_input, self.max_mse)
            self.max_mse = max_mse
            return score
        except Exception as e:
            bt.logging.error(f"Error scoring output: {e}")
            return 0.0  # Return a default score in case of an error
        
    def get_filtered_axons_from_combinations(self):
        if not self.combinations:
            self.get_filtered_axons()

        if self.combinations:
            current_combination = self.combinations.pop(0)
            bt.logging.info(f"Current Combination for VC: {current_combination}")
            filtered_axons = [self.metagraph.axons[i] for i in current_combination]
        else:
            self.get_filtered_axons()
            current_combination = self.combinations.pop(0)
            bt.logging.info(f"Current Combination for VC: {current_combination}")
            filtered_axons = [self.metagraph.axons[i] for i in current_combination]

        return filtered_axons

    def get_filtered_axons(self):
        # Get the uids of all miners in the network.
        uids = self.metagraph.uids.tolist()
        queryable_uids = (self.metagraph.total_stake >= 0)
        # Remove the weights of miners that are not queryable.
        queryable_uids = queryable_uids * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids])
        queryable_uid = queryable_uids * torch.Tensor([
            any(self.metagraph.neurons[uid].axon_info.ip == ip for ip in lib.BLACKLISTED_IPS) or
            any(self.metagraph.neurons[uid].axon_info.ip.startswith(prefix) for prefix in lib.BLACKLISTED_IPS_SEG)
            for uid in uids
        ])
        active_miners = torch.sum(queryable_uids)
        dendrites_per_query = self.total_dendrites_per_query

        # if there are no active miners, set active_miners to 1
        if active_miners == 0:
            active_miners = 1
        # if there are less than dendrites_per_query * 3 active miners, set dendrites_per_query to active_miners / 3
        if active_miners < self.total_dendrites_per_query * 3:
            dendrites_per_query = int(active_miners / 3)
        else:
            dendrites_per_query = self.total_dendrites_per_query
        
        # less than 3 set to 3
        if dendrites_per_query < self.minimum_dendrites_per_query:
                dendrites_per_query = self.minimum_dendrites_per_query
        # zip uids and queryable_uids, filter only the uids that are queryable, unzip, and get the uids
        zipped_uids = list(zip(uids, queryable_uids))
        zipped_uid = list(zip(uids, queryable_uid))
        filtered_zipped_uids = list(filter(lambda x: x[1], zipped_uids))
        filtered_uids = [item[0] for item in filtered_zipped_uids] if filtered_zipped_uids else []
        filtered_zipped_uid = list(filter(lambda x: x[1], zipped_uid))
        filtered_uid = [item[0] for item in filtered_zipped_uid] if filtered_zipped_uid else []
        self.filtered_axon = filtered_uid
        subset_length = min(dendrites_per_query, len(filtered_uids))
        # Shuffle the order of members
        random.shuffle(filtered_uids)
        # Generate subsets all items are covered
        while filtered_uids:
            subset = filtered_uids[:subset_length]
            self.combinations.append(subset)
            filtered_uids = filtered_uids[subset_length:]
        return self.combinations