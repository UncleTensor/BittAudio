# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG Team
# Copyright © 2023 <ETG>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Step 1: Import necessary libraries and modules
import os
import time
import torch
import argparse
import bittensor as bt
from datasets import load_dataset
import random
import csv
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import torchaudio
# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

# Print current working directory and directories in sys.path
print("Current working directory:", os.getcwd())
print("Directories in sys.path:", sys.path)

# Print the contents of 'AudioSubnet' directory
print("Contents of 'AudioSubnet':", os.listdir(audio_subnet_path))

# Import your module
import lib
import traceback
#Define the BittensorValidator class
class BittensorValidator:
    # List of prompts to be used for validation
    example_prompts = [
    "Can you tell me a joke that will make me laugh?",
    "Describe your favorite place to go on vacation.",
    "If you could have dinner with any historical figure, who would it be and why?",
    "What is the most challenging thing you've ever done?",
    "If you could travel to any fictional world, where would you go?",
    "Tell me about a skill or hobby you've always wanted to learn.",
    "If you were an animal, what would you be and why?",
    "What is your favorite memory from childhood?",
    "Describe a perfect day from morning to night.",
    "If you could have any superpower, what would it be?",
    "What book or movie has had the greatest impact on your life?",
    "Share a piece of advice that has stuck with you.",
    "If you could time travel, what era would you visit and why?",
    "Tell me about a goal or dream you're currently working towards.",
    "What is your favorite way to relax and unwind?",
    "If you could have a conversation with your future self, what would you ask?",
    "Describe a technology or invention you wish existed.",
    "What is your go-to comfort food?",
    "If you were a character in a fantasy novel, what role would you play?",
    "Share a travel destination on your bucket list and why you want to go there.",
    "Tell me about a moment that changed your perspective on life.",
    "What is your favorite season and why?",
    "If you could meet any celebrity, who would it be and what would you ask them?",
    "Describe a talent or skill you're proud of developing.",
    "If you could have dinner with someone from the future, who would it be?",
    "What historical event would you like to witness firsthand?",
    "Tell me about a place you consider your sanctuary or happy place.",
    "What is a hobby or activity you enjoy doing in your free time?",
    "If you could choose a new name for yourself, what would it be?",
    "Describe the perfect weekend getaway.",
    "What is the most interesting fact you've learned recently?",
    "If you could learn any language instantly, which one would it be?",
    "Tell me about a goal you've achieved that you're proud of.",
    "Describe a piece of art or music that resonates with you.",
    "If you could live in any fictional universe, where would it be?",
    "Share a personal mantra or quote that inspires you.",
    "What is your favorite type of cuisine and why?",
    "If you could have a conversation with any historical figure, who would it be?",
    "Tell me about a cultural tradition or festival you find fascinating.",
    "What is the most adventurous thing you've ever done?",
    "If you could be any character from a book or movie, who would you be?",
    "Describe a goal you have for the next year.",
    "What is your favorite way to stay active and healthy?",
    "If you could attend any major event in history, which one would it be?",
    "Tell me about a place you've visited that left a lasting impression.",
    "What is your favorite childhood memory with a sibling or friend?",
    "If you could have any animal as a pet, what would it be?",
    "Describe a dream or aspiration you have for the future.",
    "What is your favorite holiday and how do you celebrate it?",
    "If you could possess a unique talent, what would it be?",
    "Tell me about a challenge you've overcome and what you learned from it.",
    "What is a skill or hobby you'd like to learn in the next year?",
    "If you could be famous for one thing, what would it be?",
    "Describe a place you've always wanted to visit but haven't yet.",
    "What is your favorite type of outdoor activity?",
    "If you could have any job for a day, what would it be?",
    "Tell me about a lesson you've learned from a difficult experience.",
    "What is your favorite childhood game or activity?",
    "If you could visit any fictional world, where would you go?",
    "Describe a tradition or ritual that brings you joy.",
    "What is your favorite type of music or genre?",
    "If you could be a character in a fairy tale, who would you be?",
    "Tell me about a random act of kindness you've experienced.",
    "What is your favorite way to spend a lazy Sunday?",
    "If you could have dinner with any fictional character, who would it be?",
    "Describe a favorite memory from a family gathering.",
    "What is your favorite type of literature or book genre?",
    "If you could have any view from your window, what would it be?",
    "Tell me about a goal you've set for yourself in the past and achieved.",
    "What is your favorite type of dessert?",
    "If you could possess any skill instantly, what would it be?",
    "Describe a place you consider your personal oasis.",
    "What is your favorite type of movie or film genre?",
    "If you could have any animal as a companion, what would it be?",
    "Tell me about a mentor or role model who has influenced your life."]

    def __init__(self, config):
        self.config = config
        self.prompts = self._load_prompts()
        self.step = 0

    #load prompts from the dataset if the hub key is provided, otherwise use the example prompts
    def _load_prompts(self):
        if self.config.hub_key:
            gs_dev = load_dataset("speechcolab/gigaspeech", "dev", use_auth_token=self.config.hub_key)
            return gs_dev['validation']['text']
        else:
            return BittensorValidator.example_prompts
    #initialize logging for the validator 
    def _initialize_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint} with config:"
        )
        bt.logging.info(self.config)
    #initialize wallet, subtensor, dendrite, and metagraph objects 
    def _initialize_objects(self):
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
    #connect to the network and check if the wallet is registered to the chain connection
    def _connect_to_network(self):
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} if not registered to chain connection: {self.subtensor} \nRun btcli register and try again."
            )
            exit()
    #initialize weights for each dendrite in the metagraph 
    def _initialize_weights(self):
        self.scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
    #prompts will be randomly selected from the list of prompts and sent to the dendrites for processing and response generation 
    def _handle_prompt(self, step=0, current_block=None, last_updated_block=None, last_reset_weights_block=None):
        while True:
            try:
                self.metagraph.sync(subtensor=self.subtensor)
                bt.logging.info(f"🔄 Syncing metagraph with subtensor.")

                uids = self.metagraph.uids.tolist()
                if len(uids) > len(self.scores):
                    size_difference = len(uids) - len(self.scores)
                    new_scores = torch.zeros(size_difference, dtype=torch.float32)
                    self.scores = torch.cat((self.scores, new_scores))
                    del new_scores

                queryable_uids = (self.metagraph.total_stake >= 0)
                queryable_uids = queryable_uids * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids])
                active_miners = torch.sum(queryable_uids)
                dendrites_per_query = 25
                minimum_dendrites = 3

                if active_miners == 0:
                    active_miners = 1

                if active_miners < 25 * 3:
                    dendrites_per_query = int(active_miners / 3)
                else:
                    dendrites_per_query = 25

                if dendrites_per_query < minimum_dendrites:
                    dendrites_per_query = minimum_dendrites

                zipped_uids = list(zip(uids, queryable_uids))
                filtered_uids = list(zip(*filter(lambda x: x[1], zipped_uids)))[0]
                dendrites_to_query = random.sample( filtered_uids, min( dendrites_per_query, len(filtered_uids) ) )
                bt.logging.info(f"dendrites_to_query:{dendrites_to_query}")
                # Query dendrites for responses to prompts and score them using the reward module 
                try:
                    filtered_axons = [self.metagraph.axons[i] for i in dendrites_to_query]
                    if step % 1 == 0:
                        random_prompt = random.choice(self.prompts)
                        responses = self.dendrite.query(
                            filtered_axons,
                            lib.protocol.TextToSpeech(roles=["user"], text_input=random_prompt),
                            deserialize=True,
                            timeout=60,
                        )

                        for iax, resp_i in zip(filtered_axons, responses):
                            if isinstance(resp_i, lib.protocol.TextToSpeech):
                                text_input = resp_i.text_input
                                speech_output = resp_i.speech_output
                                if speech_output is not None:
                                    try:
                                        speech_tensor = torch.Tensor(speech_output)
                                        audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))
                                        audio_data_int = (audio_data * 2147483647).type(torch.IntTensor)
                                        audio_data_int = audio_data_int.unsqueeze(0)
                                        output_path = os.path.join('/tmp', f'output_{iax.hotkey}.wav')
                                        # Check if any WAV file with .wav extension exists and delete it
                                        existing_wav_files = [f for f in os.listdir('/tmp') if f.endswith('.wav')]
                                        for existing_file in existing_wav_files:
                                            try:
                                                os.remove(os.path.join('/tmp', existing_file))
                                            except Exception as e:
                                                bt.logging.error(f"Error deleting existing WAV file: {e}")
                                        if resp_i.model_name == "suno/bark":
                                            torchaudio.save(output_path, src=audio_data_int, sample_rate=24000)
                                            print(f"Saved audio file to suno/bark -----{output_path}")
                                        else:
                                            torchaudio.save(output_path, src=audio_data_int, sample_rate=16000)
                                            print(f"Saved audio file to {output_path}")
                                        score = lib.reward.score(output_path, text_input)
                                        bt.logging.info(f"Score : {score}")
                                        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                                        with open('scores.csv', 'a', newline='') as csvfile:
                                            writer = csv.writer(csvfile)
                                            writer.writerow([output_path, score, current_time])
                                        df = pd.read_csv('scores.csv')
                                        # make columns headers of df
                                        df.columns = ["Files w/ Hotkey", "Score", "Time"]
                                        # print the row if it is not empty
                                        if not df.empty:
                                            print(df.tail(1))
                                        # Delete the WAV file
                                        os.remove(output_path)
                                        zipped_uids = list(zip(uids, self.metagraph.axons))
                                        uid_index = list(zip(*filter(lambda x: x[1] == iax, zipped_uids)))[0][0]
                                        self.scores[uid_index] = self.config.alpha * self.scores[uid_index] + (1 - self.config.alpha) * score

                                    except Exception as e:
                                        bt.logging.error(f"Error writing WAV file: {e}")
                                else:
                                    bt.logging.warning(f"Received None speech_output for prompt: {text_input}. Skipping.")

                        bt.logging.info(f"Scores: {self.scores}")

                        current_block = self.subtensor.block
                        # Update weights every 100 blocks
                        if current_block - last_updated_block > 50:
                            weights = self.scores / torch.sum(self.scores)
                            bt.logging.info(f"Setting weights: {weights}")

                            processed_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
                                uids=self.metagraph.uids,
                                weights=weights,
                                netuid=self.config.netuid,
                                subtensor=self.subtensor
                            )

                            bt.logging.info(f"Processed weights: {processed_weights}")
                            bt.logging.info(f"Processed uids: {processed_uids}")
                            result = self.subtensor.set_weights(
                                netuid=self.config.netuid,
                                wallet=self.wallet,
                                uids=processed_uids,
                                weights=processed_weights,
                            )

                            last_updated_block = current_block
                            if result:
                                bt.logging.success('Successfully set weights.')
                            else:
                                bt.logging.error('Failed to set weights.')

                    step += 1
                    # Reset weights of validators and nodes without IPs every 1800 blocks
                    if last_reset_weights_block + 1800 < current_block:
                        bt.logging.trace(f"Clearing weights for validators and nodes without IPs")
                        last_reset_weights_block = current_block

                        self.scores = self.scores * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids])

                    self.metagraph = self.subtensor.metagraph(self.config.netuid)
                    time.sleep(bt.__blocktime__ * 5)

                except Exception as e:
                    bt.logging.error(f"Error querying or processing responses: {e}")
                    traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                exit()
    #run the validator 
    def run_validator(self):
        self._initialize_logging()
        self._initialize_objects()
        self._connect_to_network()
        self._initialize_weights()
        current_block = self.subtensor.block
        last_updated_block = current_block - (current_block % 100)
        last_reset_weights_block = current_block
        self._handle_prompt(step=self.step, current_block=current_block, last_updated_block=last_updated_block, last_reset_weights_block=last_reset_weights_block)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha", default=0.9, type=float, help="The weight moving average scoring."
    )
    parser.add_argument(
        "--custom", default="my_custom_value", help="Adds a custom value to the parser."
    )
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    parser.add_argument("--hub_key", type=str, default=None, help="Supply the Huggingface Hub API key for prompt dataset")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "validator",
        )
    )
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

if __name__ == "__main__":
    config = get_config()
    validator = BittensorValidator(config)
    validator.run_validator()
