
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
# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)
# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)


class MusicGenerationService(AIModelService):
    def __init__(self):
        super().__init__()  # Initializes base class components
        self.load_prompts()
        self.total_dendrites_per_query = 10
        self.minimum_dendrites_per_query = 3  # Example value, adjust as needed
        self.current_block = self.subtensor.block
        self.last_updated_block = self.current_block - (self.current_block % 100)
        self.last_reset_weights_block = self.current_block
        self.islocaltts = False
        self.p_index = 0
        self.filtered_axon = []
        self.combinations = []
        self.duration = 755  #755 tokens = 15 seconds music
        
        ###################################### DIRECTORY STRUCTURE ###########################################
        self.ttm_source_dir = os.path.join(audio_subnet_path, "ttm_source")
        # Check if the directory exists
        if not os.path.exists(self.ttm_source_dir):
            # If not, create the directory
            os.makedirs(self.ttm_source_dir)
        self.ttm_target_dir = os.path.join(audio_subnet_path, 'ttm_target')
        # Check if the directory exists
        if not os.path.exists(self.ttm_target_dir):
            # If not, create the directory
            os.makedirs(self.ttm_target_dir)
        ###################################### DIRECTORY STRUCTURE ###########################################

    def load_prompts(self):
        gs_dev = load_dataset("etechgrid/prompts_for_TTM")
        self.prompts = gs_dev['train']['text']
        return self.prompts
        
    def load_local_prompts(self):
        if os.listdir(self.ttm_source_dir):  
            self.local_prompts = pd.read_csv(os.path.join(self.ttm_source_dir, 'ttm_prompts.csv'), header=None, index_col=False)
            self.local_prompts = self.local_prompts[0].values.tolist()
            bt.logging.info(f"Loaded prompts from {self.ttm_source_dir}")
            os.remove(os.path.join(self.ttm_source_dir, 'ttm_prompts.csv'))
        
    async def run_async(self):
        step = 0

        while True:
            try:
                await self.main_loop_logic(step)
                step += 1
                await asyncio.sleep(0.5)  # Adjust the sleep time as needed
                if step % 500 == 0:
                    lib.utils.try_update()
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting MusicGenerationService.")
                break
            except Exception as e:
                print(f"An error occurred in MusicGenerationService: {e}")
                traceback.print_exc()

    async def main_loop_logic(self, step):
        # Sync and update weights logic
        if step % 5 == 0:
            self.metagraph.sync(subtensor=self.subtensor)
            bt.logging.info(f"🔄 Syncing metagraph with subtensor.")
        
        uids = self.metagraph.uids.tolist()
        # If there are more uids than scores, add more weights.
        if len(uids) > len(self.scores):
            bt.logging.trace("Adding more weights")
            size_difference = len(uids) - len(self.scores)
            new_scores = torch.zeros(size_difference, dtype=torch.float32)
            self.scores = torch.cat((self.scores, new_scores))
            del new_scores

        # check if there is a file in the tts_source directory with the name tts_prompts.csv
        if os.path.exists(os.path.join(self.ttm_source_dir, 'ttm_prompts.csv')) and not self.islocaltts:
            self.islocaltts = True
            self.load_local_prompts()
            l_prompts = self.local_prompts
            for p_index, lprompt in enumerate(l_prompts):                
                # if step % 2 == 0:
                if len(lprompt) > 256:
                    bt.logging.error(f'The length of current Prompt is greater than 256. Skipping current prompt.')
                    continue
                self.p_index = p_index
                filtered_axons = self.get_filtered_axons_from_combinations()
                bt.logging.info(f"--------------------------------- Prompt are being used locally for Text-To-Music---------------------------------")
                bt.logging.info(f"______________TTM-Prompt______________: {lprompt}")
                responses = self.query_network(filtered_axons,lprompt)
                self.process_responses(filtered_axons,responses, lprompt)

                if self.last_reset_weights_block + 1800 < self.current_block:
                    bt.logging.info(f"Clearing weights for validators and nodes without IPs")
                    self.last_reset_weights_block = self.current_block        
                    # set all nodes without ips set to 0
                    scores = scores * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in self.metagraph.uids])
            self.islocaltts = False
        else:
            g_prompts = self.load_prompts()
            g_prompt = random.choice(g_prompts)
            while len(g_prompt) > 256:
                bt.logging.error(f'The length of current Prompt is greater than 256. Skipping current prompt.')
                g_prompt = random.choice(g_prompts)
            if step % 40 == 0:
                filtered_axons = self.get_filtered_axons_from_combinations()
                bt.logging.info(f"--------------------------------- Prompt are being used from HuggingFace Dataset for Text-To-Music ---------------------------------")
                bt.logging.info(f"______________TTM-Prompt______________: {g_prompt}")
                responses = self.query_network(filtered_axons,g_prompt)
                self.process_responses(filtered_axons,responses, g_prompt)

                if self.last_reset_weights_block + 1800 < self.current_block:
                    bt.logging.info(f"Clearing weights for validators and nodes without IPs")
                    self.last_reset_weights_block = self.current_block        
                    # set all nodes without ips set to 0
                    scores = scores * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in self.metagraph.uids])

    def query_network(self,filtered_axons, prompt):
        # Network querying logic
        
        responses = self.dendrite.query(
            filtered_axons,
            lib.protocol.MusicGeneration(roles=["user"], text_input=prompt, duration=self.duration ),
            deserialize=True,
            timeout=120,
        )
        return responses
    
    def process_responses(self,filtered_axons, responses, prompt):
        for axon, response in zip(filtered_axons, responses):
            if response is not None and isinstance(response, lib.protocol.MusicGeneration):
                self.process_response(axon, response, prompt)
        
        bt.logging.info(f"Scores: {self.scores}")


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
        except Exception as e:
            bt.logging.error(f'An error occurred while handling speech output: {e}')


    def get_duration(self, wav_file_path):
        with contextlib.closing(wave.open(wav_file_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration

    def handle_music_output(self, axon, music_output, prompt, model_name):
        try:
            # Convert the list to a tensor
            speech_tensor = torch.Tensor(music_output)
            # Normalize the speech data
            audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))

            # Convert to 32-bit PCM
            audio_data_int = (audio_data * 2147483647).type(torch.IntTensor)

            # Add an extra dimension to make it a 2D tensor
            audio_data_int = audio_data_int.unsqueeze(0)

            # Save the audio data as a .wav file
            if self.islocaltts:
                output_path = os.path.join(self.ttm_target_dir, f'{self.p_index}_output_{axon.hotkey}.wav')
            else:
                # After saving the audio file
                output_path = os.path.join('/tmp', f'output_music_{axon.hotkey}.wav')
                sampling_rate = 32000
                torchaudio.save(output_path, src=audio_data_int, sample_rate=sampling_rate)
                bt.logging.info(f"Saved audio file to {output_path}")

                # Calculate the duration
                duration = self.get_duration(output_path)
                token = duration * 50.2
                bt.logging.info(f"The duration of the audio file is {duration} seconds.")
            if token < self.duration:
                bt.logging.error(f"The duration of the audio file is less than {self.duration / 50.2} seconds.Punishing the axon.")
                self.punish(axon, service="Text-To-Music", punish_message=f"The duration of the audio file is less than {self.duration / 50.2} seconds.")
                return
            else:
                # Score the output and update the weights
                score = self.score_output(output_path, prompt)
                bt.logging.info(f"Aggregated Score from Smoothness, SNR and Consistancy Metric: {score}")
                self.update_score(axon, score, service="Text-To-Music", ax=self.filtered_axon)

        except Exception as e:
            bt.logging.error(f"Error processing speech output: {e}")


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

