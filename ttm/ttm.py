from lib.hashing import load_hashes_to_cache, check_duplicate_music, save_hash_to_file
from ttm.ttm_score import MusicQualityEvaluator
from ttm.protocol import MusicGeneration
from ttm.aimodel import AIModelService
from datasets import load_dataset
from datetime import datetime
from tabulate import tabulate
import bittensor as bt
import soundfile as sf
import numpy as np
import torchaudio
import contextlib
import traceback
import asyncio
import hashlib
import random
import torch
import wandb
import wave
import lib
import sys
import os
import re


# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
audio_subnet_path = os.path.abspath(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)
refrence_dir = "audio_files"  # Directory to save the audio files
class MusicGenerationService(AIModelService):
    def __init__(self):
        super().__init__()  
        self.load_prompts()
        self.total_dendrites_per_query = 10
        self.minimum_dendrites_per_query = 3
        self.current_block = self.subtensor.block
        self.last_updated_block = self.current_block - (self.current_block % 100)
        self.last_reset_weights_block = self.current_block
        self.filtered_axon = []
        self.combinations = []
        self.duration = None
        self.lock = asyncio.Lock()
        self.audio_path = None
        # Load hashes from file to cache at startup
        load_hashes_to_cache()        

    def load_prompts(self):
        # Load the dataset (you can change this to any other dataset name)
        dataset = load_dataset("etechgrid/ttm-validation-dataset", split="train")  # Adjust the split if needed (train, test, etc.)
        random_index = random.randint(0, len(dataset) - 1)
        self.random_sample = dataset[random_index]
        # Checking if the prompt exists in the dataset
        if 'Prompts' in self.random_sample:
            prompt = self.random_sample['Prompts']
            bt.logging.info(f"Returning the prompt: {prompt}")
        else:
            print("'Prompt' not found in the sample.")
            return None  # Return None if no prompt found

        # Check if audio data exists and save it
        if 'File_Path' in self.random_sample and isinstance(self.random_sample['File_Path'], dict):
            file_path = self.random_sample['File_Path']
            if 'array' in file_path and 'sampling_rate' in file_path:
                audio_array = file_path['array']
                sample_rate = file_path['sampling_rate']

                # Save the audio to a file
                os.makedirs(refrence_dir, exist_ok=True)  # Create output directory if it doesn't exist
                audio_path = os.path.join(refrence_dir, "random_sample.wav")
                
                try:
                    # Save the audio data using soundfile
                    sf.write(audio_path, audio_array, sample_rate)
                    bt.logging.info(f"Audio saved successfully at: {audio_path}")
                    self.audio_path = audio_path
                
                    # Read the audio file into a numerical array
                    audio_data, sample_rate = sf.read(self.audio_path)
                    
                    # Convert the numerical array to a tensor
                    speech_tensor = torch.Tensor(audio_data)
                    
                    # Normalize the speech data
                    audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))
                    audio_hash = hashlib.sha256(audio_data.numpy().tobytes()).hexdigest()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Check if the music hash is a duplicate
                    if check_duplicate_music(audio_hash):
                        bt.logging.info(f"Duplicate music detected from Validator. skipping hash.")
                    else:
                        try:
                            save_hash_to_file(audio_hash, timestamp)
                            bt.logging.info(f"Music hash processed and saved successfully for Validator")
                        except Exception as e:
                            bt.logging.error(f"Error saving audio hash: {e}")
                
                except Exception as e:
                    bt.logging.error(f"Error saving audio file: {e}")
            else:
                print("Invalid audio data in 'File_Path'. Expected 'array' and 'sampling_rate'.")
                return None
        else:
            print("'File_Path' not found or invalid format in the sample.")
            return None
        
        return prompt  # Return the prompt after saving the audio file

    
    async def run_async(self):
        step = 0
        while True:
            try:
                await self.main_loop_logic(step)
                step += 1
                if step % 10 == 0:
                    self.metagraph.sync(subtensor=self.subtensor)
                    bt.logging.info(f"🔄 Syncing metagraph with subtensor.")
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting MusicGenerationService.")
                break
            except Exception as e:
                print(f"An error occurred in MusicGenerationService: {e}")
                traceback.print_exc()

    async def main_loop_logic(self, step):
        g_prompt = None
        try:
            # Load prompt from the dataset using the load_prompts function
            bt.logging.info(f"Using prompt from HuggingFace Dataset for Text-To-Music at Step: {step}")
            g_prompt = self.load_prompts()
            
            if isinstance(g_prompt, str):
                g_prompt = self.convert_numeric_values(g_prompt)

            # Ensure prompt length does not exceed 256 characters
            while isinstance(g_prompt, str) and len(g_prompt) > 256:
                bt.logging.error(f'The length of current Prompt is greater than 256. Skipping current prompt.')
                g_prompt = self.load_prompts()  # Reload another prompt
                g_prompt = self.convert_numeric_values(g_prompt)

            # Get filtered axons and query the network
            filtered_axons = self.get_filtered_axons_from_combinations()
            responses = self.query_network(filtered_axons, g_prompt)
            try:
                self.process_responses(filtered_axons, responses, g_prompt)
            except Exception as e:
                bt.logging.error(f"getting an error in processing response: {e}")

            if self.last_reset_weights_block + 50 < self.current_block:
                bt.logging.info(f"Resetting weights for validators and nodes without IPs")
                self.last_reset_weights_block = self.current_block        
                # set all nodes without ips set to 0
                self.scores = torch.Tensor(self.scores)  # Convert NumPy array to PyTorch tensor
                self.scores = self.scores * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in self.metagraph.uids])

        except Exception as e:
            bt.logging.error(f"An error occurred in main loop logic: {e}")

    def query_network(self, filtered_axons, prompt, duration=15):
        # Network querying logic
        if duration == 15:
            self.duration = 755
            self.time_out = 100
        elif duration == 30:
            self.duration = 1510
            self.time_out = 200

        """Queries the network with filtered axons and prompt."""
        responses = self.dendrite.query(
            filtered_axons,
            MusicGeneration(text_input=prompt, duration=self.duration),
            deserialize=True,
            timeout=200,
        )
        return responses
    
    def update_block(self):
        self.current_block = self.subtensor.block
        if self.current_block - self.last_updated_block > 120:
            bt.logging.info(f"Updating weights. Last update was at block: {self.last_updated_block}")
            bt.logging.info(f"Current block is for weight update is: {self.current_block}")
            self.update_weights(self.scores)
            self.last_updated_block = self.current_block
        else:
            bt.logging.info(f"Updating weights. Last update was at block:  {self.last_updated_block}")
            bt.logging.info(f"Current block is: {self.current_block}")
            bt.logging.info(f"Next update will be at block: {self.last_updated_block + 120}")
            bt.logging.info(f"Skipping weight update. Last update was at block {self.last_updated_block}")

    def process_responses(self, filtered_axons, responses, prompt):
        """Processes responses received from the network."""
        for axon, response in zip(filtered_axons, responses):
            if response is not None and isinstance(response, MusicGeneration):
                self.process_response(axon, response, prompt)
        
        bt.logging.info(f"Scores after update in TTM: {self.scores}")
        self.update_block()

    def process_response(self, axon, response, prompt, api=False):
        try:
            music_output = response.music_output
            if response is not None and isinstance(response, MusicGeneration) and response.music_output is not None and response.dendrite.status_code == 200:
                bt.logging.success(f"Received music output from {axon.hotkey}")
                if api:
                    file = self.handle_music_output(axon, music_output, prompt, response.model_name)
                    return file
                else:
                    self.handle_music_output(axon, music_output, prompt, response.model_name)
            elif response.dendrite.status_code != 403:
                self.punish(axon, service="Text-To-Music", punish_message=response.dendrite.status_message)
            else:
                pass

        except Exception as e:
            bt.logging.error(f'An error occurred while handling speech output: {e}')

    def handle_music_output(self, axon, music_output, prompt, model_name):
        # Handle the music output received from the miners
        try:
            # Convert the list to a tensor
            speech_tensor = torch.Tensor(music_output)
            bt.logging.info("Converted music output to tensor successfully.")
        except Exception as e:
            bt.logging.error(f"Error converting music output to tensor: {e}")
            return
        
        try:
            # Normalize the speech data
            audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))
            bt.logging.info("Normalized the audio data.")
        except Exception as e:
            bt.logging.error(f"Error normalizing audio data: {e}")
            return
        
        try:
            # Convert to 32-bit PCM
            audio_data_int_ = (audio_data * 2147483647).type(torch.IntTensor)
            bt.logging.info("Converted audio data to 32-bit PCM.")
        
            # Add an extra dimension to make it a 2D tensor
            audio_data_int = audio_data_int_.unsqueeze(0)
            bt.logging.info("Added an extra dimension to audio data.")
        except Exception as e:
            bt.logging.error(f"Error converting audio data to 32-bit PCM: {e}")
            return

        try:
            # Get the .wav file from the path
            file_name = os.path.basename(self.audio_path)
            bt.logging.info(f"Saving audio file to: {file_name}")
        
            # Save the audio data as a .wav file
            output_path = os.path.join('/tmp/music/', file_name)
            sampling_rate = 32000
            torchaudio.save(output_path, src=audio_data_int, sample_rate=sampling_rate)
            bt.logging.info(f"Saved audio file to {output_path}")
        except Exception as e:
            bt.logging.error(f"Error saving audio file: {e}")
            return

        try:
            # Calculate the audio hash
            audio_hash = hashlib.sha256(audio_data.numpy().tobytes()).hexdigest()
            bt.logging.info("Calculated audio hash.")
        except Exception as e:
            bt.logging.error(f"Error calculating audio hash: {e}")
            return
        
        try:
            # Check if the music hash is a duplicate
            if check_duplicate_music(audio_hash):
                bt.logging.info(f"Duplicate music detected from miner: {axon.hotkey}. Issuing punishment.")
                self.punish(axon, service="Text-To-Music", punish_message="Duplicate music detected")
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_hash_to_file(audio_hash, axon.hotkey, timestamp)
                bt.logging.info(f"Music hash processed and saved successfully for miner: {axon.hotkey}")
        except Exception as e:
            bt.logging.error(f"Error checking or saving music hash: {e}")
            return
        
        try:
            # Log the audio to wandb
            uid_in_metagraph = self.metagraph.hotkeys.index(axon.hotkey)
            audio_data_np = np.array(audio_data_int_)
            wandb.log({
                f"TTM prompt: {prompt[:100]} ....": wandb.Audio(audio_data_np, caption=f'For HotKey: {axon.hotkey[:10]} and uid {uid_in_metagraph}', sample_rate=sampling_rate)
            })
            bt.logging.success(f"TTM Audio file uploaded to wandb successfully for Hotkey {axon.hotkey} and UID {uid_in_metagraph}")
        except Exception as e:
            bt.logging.error(f"Error uploading TTM audio file to wandb: {e}")
        
        try:
            # Get audio duration
            duration = self.get_duration(output_path)
            token = duration * 50.2
            bt.logging.info(f"The duration of the audio file is {duration} seconds.")
        except Exception as e:
            bt.logging.error(f"Error calculating audio duration: {e}")
            return

        try:
            refrence_dir = self.audio_path
            score, table1, table2 = self.score_output("/tmp/music/", refrence_dir, prompt)
            if duration < 15:
                score = self.score_adjustment(score, duration)
                bt.logging.info(f"Score updated based on short duration than required: {score}")
            else:
                bt.logging.info(f"Duration is greater than 15 seconds. No need to penalize the score.")
        except Exception as e:
            bt.logging.error(f"Error scoring the output: {e}")
            return

        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tabulated_str = tabulate(table1, headers=[f"Raw score for hotkey:{axon.hotkey}", current_datetime], tablefmt="grid")
            print(tabulated_str)
            print("\n")
            tabulated_str2 = tabulate(table2, headers=[f"Normalized score for hotkey:{axon.hotkey}", current_datetime], tablefmt="grid")
            print(tabulated_str2)
            bt.logging.info(f"Aggregated Score for hotkey {axon.hotkey}: {score}")
            self.update_score(axon, score, service="Text-To-Music")
        except Exception as e:
            bt.logging.error(f"Error generating score tables or updating score: {e}")
            return
        
        return output_path


    def get_duration(self, wav_file_path):
        """Returns the duration of the audio file in seconds."""
        with contextlib.closing(wave.open(wav_file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)

    def score_adjustment(self, score, duration):
        """Adjusts the score based on the duration of the generated audio."""
        conditions = [
            (lambda d: 14.5 <= d < 15, 0.9),
            (lambda d: 14 <= d < 14.5, 0.8),
            (lambda d: 13.5 <= d < 14, 0.7),
            (lambda d: 13 <= d < 13.5, 0.6),
            (lambda d: 12.5 <= d < 13, 0.0),
        ]
        for condition, multiplier in conditions:
            if condition(duration):
                return score * multiplier
        return score

    def score_output(self, output_path, refrence_dir , prompt):
        """Evaluates and returns the score for the generated music output."""
        try:
            score_object = MusicQualityEvaluator()
            return score_object.evaluate_music_quality(output_path, refrence_dir, prompt)
        except Exception as e:
            bt.logging.error(f"Error scoring output: {e}")
            return 0.0


    def get_filtered_axons_from_combinations(self):
        if not self.combinations:
            self.get_filtered_axons()

        if self.combinations:
            current_combination = self.combinations.pop(0)
            bt.logging.info(f"Current Combination for TTM: {current_combination}")
            filtered_axons = [self.metagraph.axons[i] for i in current_combination]
        else:
            self.get_filtered_axons()
            current_combination = self.combinations.pop(0)
            bt.logging.info(f"Current Combination for TTM: {current_combination}")
            filtered_axons = [self.metagraph.axons[i] for i in current_combination]

        return filtered_axons
    

    def get_filtered_axons(self):
        # Get the uids of all miners in the network.
        uids = self.metagraph.uids.tolist()
        queryable_uids = (self.metagraph.total_stake >= 0)
        # Remove the weights of miners that are not queryable.
        queryable_uids = torch.Tensor(queryable_uids) * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids])

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
        filtered_zipped_uids = list(filter(lambda x: x[1], zipped_uids))
        filtered_uids = [item[0] for item in filtered_zipped_uids] if filtered_zipped_uids else []
        subset_length = min(dendrites_per_query, len(filtered_uids))
        # Shuffle the order of members
        random.shuffle(filtered_uids)
        # Generate subsets of length 7 until all items are covered
        while filtered_uids:
            subset = filtered_uids[:subset_length]
            self.combinations.append(subset)
            filtered_uids = filtered_uids[subset_length:]
        return filtered_uids #self.combinations

    def update_weights(self, scores):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners.
        The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Convert scores to a PyTorch tensor and check for NaN values
        weights = torch.tensor(scores)
        if torch.isnan(weights).any():
            bt.logging.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Normalize scores to get raw weights
        raw_weights = torch.nn.functional.normalize(weights, p=1, dim=0)
        bt.logging.info("raw_weights", raw_weights)

        # Convert uids to a PyTorch tensor
        uids = torch.tensor(self.metagraph.uids)

        bt.logging.info("raw_weight_uids", uids)

        try:
            # Convert tensors to NumPy arrays for processing if required by the process_weights_for_netuid function
            uids_np = uids.numpy() if isinstance(uids, torch.Tensor) else uids
            raw_weights_np = raw_weights.numpy() if isinstance(raw_weights, torch.Tensor) else raw_weights

            # Process the raw weights and uids based on subnet limitations
            (processed_weight_uids, processed_weights) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=uids_np,  # Ensure this is a NumPy array
                weights=raw_weights_np,  # Ensure this is a NumPy array
                netuid=self.config.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )
            bt.logging.info("processed_weights", processed_weights)
            bt.logging.info("processed_weight_uids", processed_weight_uids)
        except Exception as e:
            bt.logging.error(f"An error occurred while processing weights within update_weights: {e}")
            return

        # Convert processed weights and uids back to PyTorch tensors if needed for further processing
        processed_weight_uids = torch.tensor(processed_weight_uids) if isinstance(processed_weight_uids, np.ndarray) else processed_weight_uids
        processed_weights = torch.tensor(processed_weights) if isinstance(processed_weights, np.ndarray) else processed_weights

        # Convert weights and uids to uint16 format for emission
        uint_uids, uint_weights = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.info("uint_weights", uint_weights)
        bt.logging.info("uint_uids", uint_uids)

        # Set the weights on the Bittensor network
        try:
            result, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=self.version,
            )

            if result:
                bt.logging.info("Weights set on the chain successfully!")
            else:
                bt.logging.error(f"Failed to set weights: {msg}")
        except Exception as e:
            bt.logging.error(f"An error occurred while setting weights: {e}")
