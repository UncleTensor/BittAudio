
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
import lib
import traceback
import pandas as pd
import sys
import wandb
import datetime as dt
import numpy as np
# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)
# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)


class TextToSpeechService(AIModelService):
    def __init__(self):
        super().__init__()  # Initializes base class components
        self.load_prompts()
        self.total_dendrites_per_query = 15
        self.minimum_dendrites_per_query = 3  # Example value, adjust as needed
        self.current_block = self.subtensor.block
        self.filtered_axon = []
        self.last_updated_block = self.current_block - (self.current_block % 100)
        self.last_reset_weights_block = self.current_block
        self.p_index = 0
        self.last_run_start_time = dt.datetime.now()
        self.tao = self.metagraph.neurons[self.uid].stake.tao
        self.combinations = []
        self.lock = asyncio.Lock()
        self.response = None
        
    def load_prompts(self):
        gs_dev = load_dataset("etechgrid/Prompts_for_Voice_cloning_and_TTS")
        self.prompts = gs_dev['train']['text']
        return self.prompts
        
        
    def check_and_update_wandb_run(self):
        # Calculate the time difference between now and the last run start time
        current_time = dt.datetime.now()
        time_diff = current_time - self.last_run_start_time
        # Check if 4 hours have passed since the last run start time
        if time_diff.total_seconds() >= 4 * 3600:  # 4 hours * 3600 seconds/hour
            self.last_run_start_time = current_time  # Update the last run start time to now
            if self.wandb_run:
                wandb.finish()  # End the current run
            self.new_wandb_run()  # Start a new run

    def new_wandb_run(self):
        now = dt.datetime.now()
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = f"Validator-{self.uid}-{run_id}"
        commit = self.get_git_commit_hash()
        self.wandb_run = wandb.init(
            name=name,
            project="AudioSubnet_Valid",
            entity="subnet16team",
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "Validator",
                "tao (stake)": self.tao,
                "commit": commit,
            },
            tags=self.sys_info,
            allow_val_change=True,
            anonymous="allow",
        )
        bt.logging.debug(f"Started a new wandb run: {name}")

    async def run_async(self):
        step = 0

        while True:
            self.check_and_update_wandb_run()
            try:
                await self.main_loop_logic(step)
                step += 1
                await asyncio.sleep(0.5)  # Adjust the sleep time as needed
                if step % 50 == 0 and self.config.auto_update == 'yes':
                    lib.utils.try_update()
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting TextToSpeechService.")
                break
            except Exception as e:
                print(f"An error occurred in TextToSpeechService: {e}")
                traceback.print_exc()

    async def main_loop_logic(self, step):
        # Sync and update weights logic
        if step % 10 == 0:
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
        g_prompts = self.load_prompts()
        g_prompt = random.choice(g_prompts)
        while len(g_prompt) > 256:
            bt.logging.error(f'The length of current Prompt is greater than 256. Skipping current prompt.')
            g_prompt = random.choice(g_prompts)
        if step % 20 == 0:
            async with self.lock:
                filtered_axons = self.get_filtered_axons_from_combinations()
                bt.logging.info(f"Prompt are being used from HuggingFace Dataset for TTS at Step: {step}")
                bt.logging.info(f"______________Prompt______________: {g_prompt}")
                responses = self.query_network(filtered_axons, g_prompt)
                self.process_responses(filtered_axons, responses, g_prompt)

                if self.last_reset_weights_block + 1800 < self.current_block:
                    bt.logging.trace(f"Clearing weights for validators and nodes without IPs")
                    self.last_reset_weights_block = self.current_block        
                    # set all nodes without ips set to 0
                    self.scores = self.scores * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in self.metagraph.uids])
    def query_network(self,filtered_axons, prompt):
        # Network querying logic
        
        responses = self.dendrite.query(
            filtered_axons,
            lib.protocol.TextToSpeech(text_input=prompt),
            deserialize=True,
            timeout=50,
        )
        return responses
    
    def update_block(self):
        self.current_block = self.subtensor.block
        if self.current_block - self.last_updated_block > 100:
            bt.logging.info(f"Updating weights. Last update was at block {self.last_updated_block}")
            bt.logging.info(f"Current block is {self.current_block}")
            self.update_weights(self.scores)
            self.last_updated_block = self.current_block
        else:
            bt.logging.info(f"Updating weights. Last update was at block:  {self.last_updated_block}")
            bt.logging.info(f"Current block is: {self.current_block}")
            bt.logging.info(f"Next update will be at block: {self.last_updated_block + 100}")
            bt.logging.info(f"Skipping weight update. Last update was at block {self.last_updated_block}")

    def process_responses(self,filtered_axons, responses, prompt):
        for axon, response in zip(filtered_axons, responses):
            if response is not None and isinstance(response, lib.protocol.TextToSpeech):
                self.response = response
                self.process_response(axon, response, prompt)
        
        bt.logging.info(f"Scores after update in TTS: {self.scores}")
        self.update_block()


    def process_response(self, axon, response, prompt):
        try:
            speech_output = response.speech_output
            if response is not None and isinstance(response, lib.protocol.TextToSpeech) and response.speech_output is not None and response.dendrite.status_code == 200:
                bt.logging.success(f"Received Text to speech output from {axon.hotkey}")
                self.handle_speech_output(axon, speech_output, prompt, response.model_name)
            elif response.dendrite.status_code != 403:
                self.punish(axon, service="Text-To-Speech", punish_message=response.dendrite.status_message)
            else:
                pass
        except Exception as e:
            bt.logging.error(f'An error occurred while handling speech output: {e}')

    def handle_speech_output(self, axon, speech_output, prompt, model_name):
        try:
            # Convert the list to a tensor
            speech_tensor = torch.Tensor(speech_output)
            # Normalize the speech data
            audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))

            # Convert to 32-bit PCM
            audio_data_int_ = (audio_data * 2147483647).type(torch.IntTensor)

            # Add an extra dimension to make it a 2D tensor
            audio_data_int = audio_data_int_.unsqueeze(0)

            # Save the audio data as a .wav file
            output_path = os.path.join('/tmp', f'output_{axon.hotkey}.wav')
            
            # Check if any WAV file with .wav extension exists and delete it
            existing_wav_files = [f for f in os.listdir('/tmp') if f.endswith('.wav')]
            for existing_file in existing_wav_files:
                try:
                    os.remove(os.path.join('/tmp', existing_file))
                except Exception as e:
                    bt.logging.error(f"Error deleting existing WAV file: {e}")

            # Save the audio file
            if model_name == "suno/bark":
                sampling_rate = 24000 
            elif model_name == "elevenlabs/eleven": 
                sampling_rate = 44000
            else:
                sampling_rate = 16000
            torchaudio.save(output_path, src=audio_data_int, sample_rate=sampling_rate)
            print(f"Saved audio file to {output_path}")
            try:
                uid_in_metagraph = self.metagraph.hotkeys.index(axon.hotkey)
                wandb.log({f"Text to Speech prompt:{self.response.text_input} ": wandb.Audio(np.array(audio_data_int_), caption=f'For HotKey: {axon.hotkey[:10]} and uid {uid_in_metagraph}', sample_rate=sampling_rate)})
                bt.logging.success(f"TTS Audio file uploaded to wandb successfully for Hotkey {axon.hotkey}")
            except Exception as e:
                bt.logging.error(f"Error uploading TTS audio to wandb for Hotkey {axon.hotkey}: {e}")
            # Score the output and update the weights
            score = self.score_output(output_path, prompt)
            bt.logging.info(f"Aggregated Score from the NISQA and WER Metric: {score}")
            self.update_score(axon, score, service="Text-To-Speech", ax=self.filtered_axon)

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
            # Call the scoring function from lib.reward
            score = lib.reward.score(output_path, prompt)
            return score
        except Exception as e:
            bt.logging.error(f"Error scoring output: {e}")
            return 0.0  # Return a default score in case of an error
        
    def get_filtered_axons_from_combinations(self):
        if not self.combinations:
            self.get_filtered_axons()

        if self.combinations:
            current_combination = self.combinations.pop(0)
            bt.logging.info(f"Current Combination for TTS: {current_combination}")
            filtered_axons = [self.metagraph.axons[i] for i in current_combination]
        else:
            self.get_filtered_axons()
            current_combination = self.combinations.pop(0)
            bt.logging.info(f"Current Combination for TTS: {current_combination}")
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
    
    def update_weights(self, scores):
        # Process scores for blacklisted miners
        for idx, uid in enumerate(self.metagraph.uids):
            neuron = self.metagraph.neurons[uid]
            if neuron.coldkey in lib.BLACKLISTED_MINER_COLDKEYS or neuron.hotkey in lib.BLACKLISTED_MINER_HOTKEYS:
                scores[idx] = 0.0
                bt.logging.info(f"Blacklisted miner detected: {uid}. Score set to 0.")

        # Normalize scores to get weights
        weights = torch.nn.functional.normalize(scores, p=1, dim=0)
        bt.logging.info(f"Setting weights: {weights}")

        # Process weights for the subnet
        try:
            processed_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=weights,
                netuid=self.config.netuid,
                subtensor=self.subtensor
            )
            bt.logging.info(f"Processed weights: {processed_weights}")
            bt.logging.info(f"Processed uids: {processed_uids}")
        except Exception as e:
            bt.logging.error(f"An error occurred While processing the weights: {e}")

        try:
            # Set weights on the Bittensor network
            result = self.subtensor.set_weights(
                netuid=self.config.netuid,  # Subnet to set weights on
                wallet=self.wallet,         # Wallet to sign set weights using hotkey
                uids=processed_uids,        # Uids of the miners to set weights for
                weights=processed_weights, # Weights to set for the miners
                wait_for_finalization=True,   
            )

            if result:
                bt.logging.success(f'Successfully set weights. result: {result}')
                bt.logging.info(f'META GRPAH: {self.metagraph.E.numpy()}')
            else:
                bt.logging.error('Failed to set weights.')
        except Exception as e:
            bt.logging.error(f"An error occurred while setting weights: {e}")

