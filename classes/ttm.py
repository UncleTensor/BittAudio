
from classes.aimodel import AIModelService
import os
import bittensor as bt
import asyncio
import traceback
from datasets import load_dataset
import torch
import random
import torchaudio
# Import your module
import lib.utils
import lib.ttm_score
import lib.protocol
import lib
import traceback
import pandas as pd
import sys
import wave
import contextlib
import numpy as np
import wandb
# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)
# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)


class MusicGenerationService(AIModelService):
    def __init__(self):
        super().__init__()  
        self.load_prompts()
        self.total_dendrites_per_query = 10
        self.minimum_dendrites_per_query = 3  # Example value, adjust as needed
        self.current_block = self.subtensor.block
        self.last_updated_block = self.current_block - (self.current_block % 100)
        self.last_reset_weights_block = self.current_block
        self.filtered_axon = []
        self.combinations = []
        self.duration = 755  #755 tokens = 15 seconds music
        self.lock = asyncio.Lock()
        

    def load_prompts(self):
        gs_dev = load_dataset("etechgrid/prompts_for_TTM")
        self.prompts = gs_dev['train']['text']
        return self.prompts
        
    async def run_async(self):
        step = 0
        while self.service_flags["MusicGenerationService"]:
            try:
                await self.main_loop_logic(step)
                step += 1
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting MusicGenerationService.")
                break
            except Exception as e:
                print(f"An error occurred in MusicGenerationService: {e}")
                traceback.print_exc()

    async def main_loop_logic(self, step):
        g_prompt = None
        try:
            c_prompt = self.api.get_TTM()
        except Exception as e:
            bt.logging.error(f"An error occurred while fetching prompt: {e}")
            c_prompt = None

        if step:
            async with self.lock:
                # Use the API prompt if available; otherwise, load prompts from HuggingFace
                if c_prompt:
                    bt.logging.info(f"--------------------------------- Prompt are being used from Corcel API for Text-To-Music at Step: {step} --------------------------------- ")
                    g_prompt = self.convert_numeric_values(c_prompt)  # Use the prompt from the API
                else:
                    # Fetch prompts from HuggingFace if API failed
                    bt.logging.info(f"--------------------------------- Prompt are being used from HuggingFace Dataset for Text-To-Music at Step: {step} --------------------------------- ")
                    g_prompt = self.load_prompts()
                    g_prompt = random.choice(g_prompt)  # Choose a random prompt from HuggingFace
                    g_prompt = self.convert_numeric_values(g_prompt)

                while len(g_prompt) > 256:
                    bt.logging.error(f'The length of current Prompt is greater than 256. Skipping current prompt.')
                    g_prompt = random.choice(g_prompt)
                    g_prompt = self.convert_numeric_values(g_prompt)

                filtered_axons = self.get_filtered_axons_from_combinations()
                bt.logging.info(f"______________TTM-Prompt______________: {g_prompt}")
                responses = self.query_network(filtered_axons,g_prompt)
                self.process_responses(filtered_axons,responses, g_prompt)

    def query_network(self,filtered_axons, prompt):
        # Network querying logic
        
        responses = self.dendrite.query(
            filtered_axons,
            lib.protocol.MusicGeneration(text_input=prompt, duration=self.duration ),
            deserialize=True,
            timeout=140,
        )
        return responses
    
    def process_responses(self,filtered_axons, responses, prompt):
        for axon, response in zip(filtered_axons, responses):
            if response is not None and isinstance(response, lib.protocol.MusicGeneration):
                self.process_response(axon, response, prompt)
        
        bt.logging.info(f"Scores after update in TTM: {self.scores}")


    def process_response(self, axon, response, prompt):
        try:
            music_output = response.music_output
            if response is not None and isinstance(response, lib.protocol.MusicGeneration) and response.music_output is not None and response.dendrite.status_code == 200:
                bt.logging.success(f"Received music output from {axon.hotkey}")
                self.handle_music_output(axon, music_output, prompt, response.model_name)
            elif response.dendrite.status_code != 403:
                self.punish(axon, service="Text-To-Music", punish_message=response.dendrite.status_message)
            else:
                pass
            self.service_flags["MusicGenerationService"] = False
            self.service_flags["VoiceCloningService"] = True
        except Exception as e:
            bt.logging.error(f'An error occurred while handling speech output: {e}')


    def get_duration(self, wav_file_path):
        with contextlib.closing(wave.open(wav_file_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
        
    def score_adjustment(self, score, duration):
        conditions = [
            (lambda d: 14.5 <= d < 15, 0.9),
            (lambda d: 14 <= d < 14.5, 0.8),
            (lambda d: 13.5 <= d < 14, 0.7),
            (lambda d: 13 <= d < 13.5, 0.6),
            (lambda d: 12.5 <= d < 13, 0.0),
            # (lambda d: d >= 15, 1.0)  # Add this line
        ]
        for condition, multiplier in conditions:
            if condition(duration):
                return score * multiplier
        return score  # If none of the conditions were met


    def handle_music_output(self, axon, music_output, prompt, model_name):
        token = 0
        try:
            # Convert the list to a tensor
            speech_tensor = torch.Tensor(music_output)
            # Normalize the speech data
            audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))

            # Convert to 32-bit PCM
            audio_data_int_ = (audio_data * 2147483647).type(torch.IntTensor)

            # Add an extra dimension to make it a 2D tensor
            audio_data_int = audio_data_int_.unsqueeze(0)

            # Save the audio data as a .wav file
            # After saving the audio file
            output_path = os.path.join('/tmp', f'output_music_{axon.hotkey}.wav')
            sampling_rate = 32000
            torchaudio.save(output_path, src=audio_data_int, sample_rate=sampling_rate)
            bt.logging.info(f"Saved audio file to {output_path}")

            try:
                uid_in_metagraph = self.metagraph.hotkeys.index(axon.hotkey)
                wandb.log({f"TTM prompt: {prompt}": wandb.Audio(np.array(audio_data_int_), caption=f'For HotKey: {axon.hotkey[:10]} and uid {uid_in_metagraph}', sample_rate=sampling_rate)})
                bt.logging.success(f"TTM Audio file uploaded to wandb successfully for Hotkey {axon.hotkey} and UID {uid_in_metagraph}")
            except Exception as e:
                bt.logging.error(f"Error uploading TTM audio file to wandb: {e}")

            # Calculate the duration
            duration = self.get_duration(output_path)
            token = duration * 50.2
            bt.logging.info(f"The duration of the audio file is {duration} seconds.")
            # Score the output and update the weights
            score = self.score_output(output_path, prompt)
            bt.logging.info(f"Score output after analysing the output file: {score}")
            try:
                if duration < 15:
                    score = self.score_adjustment(score, duration)
                    bt.logging.info(f"Score updated based on short duration than the required by the client: {score}")
                else:
                    bt.logging.info(f"Duration is greater than 15 seconds. No need to penalize the score.")
            except Exception as e:
                bt.logging.error(f"Error in penalizing the score: {e}")
            bt.logging.info(f"Aggregated Score from Smoothness, SNR and Consistancy Metric: {score}")
            self.update_score(axon, score, service="Text-To-Music", ax=self.filtered_axon)
            # return output_path

        except Exception as e:
            bt.logging.error(f"Error processing Music output: {e}")


    def score_output(self, output_path, prompt):
        """
        Calculate a score for the output audio file based on the given prompt.

        Parameters:
        output_path (str): Path to the output audio file.
        prompt (str): The input prompt used to generate the speech output.

        Returns:
        float: The calculated score.
        """
        try:
            score_object = lib.ttm_score.MusicQualityEvaluator()
            # Call the scoring function from lib.reward
            score = score_object.evaluate_music_quality(output_path, prompt)
            return score
        except Exception as e:
            bt.logging.error(f"Error scoring output: {e}")
            return 0.0  # Return a default score in case of an error
        
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
        # Generate subsets of length 7 until all items are covered
        while filtered_uids:
            subset = filtered_uids[:subset_length]
            self.combinations.append(subset)
            filtered_uids = filtered_uids[subset_length:]
        return self.combinations

