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

# Determine variable state directory
audio_subnet_vardir = os.getenv("AUDIOSUBNET_VARDIR")

# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)
audio_file_path = os.path.join(audio_subnet_vardir or audio_subnet_path, "input_file.wav")
# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)


class VoiceCloningService(AIModelService):
    # Existing __init__ and other methods...
    def __init__(self):
        super().__init__()  # This is the new way, using the singleton instance
        self.load_vc_prompts()
        self.load_vc_voices()
        self.total_dendrites_per_query = self.vcdnp  # Example value, adjust as needed
        self.minimum_dendrites_per_query = 5  # Example value, adjust as needed
        self.combinations = []
        self.lock = asyncio.Lock()
        # self.best_uid = self.priority_uids(self.metagraph)
        self.filtered_axon = []
        self.filtered_axons = []
        self.responses = None
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
        while self.service_flags["VoiceCloningService"]:
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
        try:
            c_prompt = self.api.get_VC()
        except Exception as e:
            bt.logging.error(f"An error occurred while fetching prompt: {e}")
            c_prompt = None
        if step % 300 == 0:
            async with self.lock:
                if c_prompt:
                    bt.logging.info(f"--------------------------------- Prompt and voices are being used from Corcel API for Voice Clone ---------------------------------")
                    self.text_input = self.convert_numeric_values(c_prompt)  # Use the prompt from the API
                    bt.logging.info(f"______________VC-Prompt coming from Corcel______________: {self.text_input}")
                    if len(c_prompt) > 256:
                        pass
                else:
                    # Fetch prompts from HuggingFace if API failed
                    bt.logging.info(f"--------------------------------- Prompt and voices are being used from HuggingFace Dataset for Voice Clone ---------------------------------")
                    self.text_input = random.choice(self.prompts)
                    self.text_input = self.convert_numeric_values(self.text_input)
                while len(self.text_input) > 256:
                    bt.logging.error(f"The length of current Prompt is greater than 256. Skipping current prompt.")
                    self.text_input = random.choice(self.prompts)
                    self.text_input = self.convert_numeric_values(self.text_input)

                bt.logging.info(f"______________VC-Prompt______________: {self.text_input}")
                vc_voice = random.choice(self.audio_files)
                audio_array = vc_voice['array']
                sampling_rate = vc_voice['sampling_rate']
                self.hf_voice_id = vc_voice['path'].split("/")[-1][:10]
                sf.write(audio_file_path, audio_array, sampling_rate)
                self.audio_file_path = audio_file_path
                waveform, _ = torchaudio.load(self.audio_file_path)
                clone_input = waveform.tolist()
                sample_rate = sampling_rate
                await self.generate_voice_clone(self.text_input, clone_input, sample_rate)

    async def main_loop_logic(self, step):
        tasks = []
        try:
            huggingface_task = asyncio.create_task(self.process_huggingface_prompts(step))
            tasks.extend([huggingface_task ]) #local_files_task
        except Exception as e:
            bt.logging.error(f"An error occurred in VoiceCloningService: {e}")
            traceback.print_exc()

        await asyncio.sleep(0.5)  # Delay at the end of each loop iteration
        return tasks

    def read_audio_file(self, path):
        try:
            # Read the audio file and return its content
            waveform, sample_rate = torchaudio.load(path)
            return waveform, sample_rate
        except Exception as e:
            print(f"An error occurred while reading the audio file: {e}")
    
    async def generate_voice_clone(self, text_input, clone_input, sample_rate, api_axon=None, input_file=None):
        try:
            filtered_axons = api_axon if api_axon else self.get_filtered_axons_from_combinations() 
            # for ax in self.filtered_axons:
            self.responses = self.dendrite.query(
                filtered_axons,
                lib.protocol.VoiceClone(text_input=text_input, clone_input=clone_input, sample_rate=sample_rate, hf_voice_id="name"), 
                deserialize=True,
                timeout=150,
            )
            # Process the responses if needed
            processed_vc_file = self.process_voice_clone_responses(filtered_axons, text_input, input_file)
            bt.logging.info(f"Updated Scores for Voice Cloning: {self.scores}")
            self.service_flags["VoiceCloningService"] = False
            self.service_flags["TextToSpeechService"] = True
            return processed_vc_file
        except Exception as e:
            print(f"An error occurred while processing the voice clone: {e}")

    def process_voice_clone_responses(self,filtered_axons, text_input, input_file=None):
        try:
            for axon, response in zip(filtered_axons, self.responses):
                if response is not None and isinstance(response, lib.protocol.VoiceClone) and response.clone_output is not None and response.dendrite.status_code == 200:
                    bt.logging.success(f"Received Voice Clone output from {axon.hotkey}")
                    self.handle_clone_output(response, axon,  prompt=text_input, input_file=input_file)
                    # vc_file = self.handle_clone_output(response, axon,  prompt=text_input, input_file=input_file)
                    # return vc_file
                elif response.dendrite.status_code != 403:
                    self.punish(axon, service="Voice Cloning", punish_message=response.dendrite.status_message)
                else:
                    pass
        except Exception as e:
            print(f"An error occurred while processing voice clone responses: {e}")

    def handle_clone_output(self, response, axon, prompt=None, input_file=None):
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
                if input_file:
                    cloned_file_path = os.path.join('/tmp', 'API_cloned_'+ axon.hotkey[:] +'.wav' )
                    torchaudio.save(cloned_file_path, src=audio_data_int, sample_rate=sampling_rate)
                    score = self.score_output(input_file, cloned_file_path, prompt) # self.audio_file_path
                    bt.logging.info(f"The cloned file for API have been saved successfully: {cloned_file_path}")
                    try:
                        uid_in_metagraph = self.metagraph.hotkeys.index(axon.hotkey)
                        wandb.log({f"Voice Clone Prompt: {prompt}": wandb.Audio(np.array(audio_data_int_), caption=f'For HotKey: {axon.hotkey[:10]} and uid {uid_in_metagraph}', sample_rate=sampling_rate)})
                        bt.logging.success(f"Voice Clone Audio file uploaded to wandb successfully for Hotkey {axon.hotkey} and uid {uid_in_metagraph}")
                    except Exception as e:
                        bt.logging.error(f"Error uploading Voice Clone Audio file to wandb: {e}")
                    return cloned_file_path
                else:
                    cloned_file_path = os.path.join('/tmp', '_cloned_'+ axon.hotkey[:] +'.wav' )
                    torchaudio.save(cloned_file_path, src=audio_data_int, sample_rate=sampling_rate)
                    score = self.score_output(self.audio_file_path, cloned_file_path, prompt)
                    bt.logging.info(f"The cloned file have been saved successfully: {cloned_file_path}")
                    try:
                        uid_in_metagraph = self.metagraph.hotkeys.index(axon.hotkey)
                        wandb.log({f"Voice Clone Prompt: {response.text_input}": wandb.Audio(np.array(audio_data_int_), caption=f'For HotKey: {axon.hotkey[:10]} and uid {uid_in_metagraph}', sample_rate=sampling_rate)})
                        bt.logging.success(f"Voice Clone Audio file uploaded to wandb successfully for Hotkey {axon.hotkey} and uid {uid_in_metagraph}")
                    except Exception as e:
                        bt.logging.error(f"Error uploading Voice Clone Audio file to wandb: {e}")
                bt.logging.info(f"The score of the cloned file : {score}")
                self.update_score(axon, score, service="Voice Cloning") #, ax=self.filtered_axon
        except Exception as e:
            pass


    def score_output(self, input_path, output_path, text_input):
        '''Score the output based on the input and output paths'''
        try:
            clone_score = CloneScore()
            # Call the scoring function from lib.reward
            score = clone_score.compare_audio(input_path , output_path, text_input)
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
