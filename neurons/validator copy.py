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


# Base Class
import os
import argparse
import bittensor as bt
import sys
import asyncio
import traceback
from datasets import load_dataset
import torch
import random
import csv
import pandas as pd
import torchaudio
from tabulate import tabulate
# Import your module
import lib.utils
import lib
import traceback

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




class AIModelService:
    def __init__(self):
        self.config = self.get_config()
        self.setup_paths()
        self.setup_logging()
        self.setup_wallet()
        self.setup_subtensor()
        self.setup_dendrite()
        self.setup_metagraph()
        self.scores = torch.zeros_like(self.metagraph.S, dtype=torch.float32)

    def get_config(self):
        parser = argparse.ArgumentParser()

        # Add arguments as per your original script
        parser.add_argument("--alpha", default=0.9, type=float, help="The weight moving average scoring.")
        parser.add_argument("--custom", default="my_custom_value", help="Adds a custom value to the parser.")
        parser.add_argument("--auto_update", default="yes", help="Auto update")
        parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
        parser.add_argument("--hub_key", type=str, default=None, help="Supply the Huggingface Hub API key for prompt dataset")

        # Add Bittensor specific arguments
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)

        # Parse and return the config
        config = bt.config(parser)
        return config

    def setup_paths(self):
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

    def setup_logging(self):
        # Set up logging with the provided configuration and directory
        self.config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                self.config.logging.logging_dir,
                self.config.wallet.name,
                self.config.wallet.hotkey,
                self.config.netuid,
                "validator",
            )
        )
        
        # Ensure the logging directory exists
        if not os.path.exists(self.config.full_path):
            os.makedirs(self.config.full_path, exist_ok=True)

        bt.logging(self.config, logging_dir=self.config.full_path)

    def setup_wallet(self):
        # Initialize the wallet with the provided configuration
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")


    def setup_subtensor(self):
    # Initialize the subtensor connection with the provided configuration
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

    def setup_dendrite(self):
        # Initialize the dendrite (RPC client) with the wallet
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

    def setup_metagraph(self):
        # Initialize the metagraph for the network state
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

    
    async def run_async(self):
        raise NotImplementedError

class VoiceCloningService(AIModelService):
    def __init__(self):
        super().__init__()  # Initializes base class components
        self.audio_path = os.path.join(os.path.dirname(__file__), "source_audio")
        # Check if the directory exists
        if not os.path.exists(self.audio_path):
            # Create the directory if it does not exist
            os.makedirs(self.audio_path)
        self.target_audio_path = os.path.join(os.path.dirname(__file__), "target_audio")
        # Check if the directory exists
        if not os.path.exists(self.target_audio_path):
            # Create the directory if it does not exist
            os.makedirs(self.target_audio_path)
        self.processed_audio_path = os.path.join(os.path.dirname(__file__), "processed_audio")
        # Check if the directory exists
        if not os.path.exists(self.processed_audio_path):
            # Create the directory if it does not exist
            os.makedirs(self.processed_audio_path)
        self.total_dendrites_per_query = 25
        self.minimum_dendrites_per_query = 3
        self.dendrite = bt.dendrite(wallet=self.wallet)


    async def run_async(self):
        while True:
            try:
                await self.main_loop_logic(self.scores)
                await asyncio.sleep(1)  # Adjust the sleep time as needed
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting VoiceCloningService.")
                break
            except Exception as e:
                print(f"An error occurred in VoiceCloningService: {e}")
                traceback.print_exc()

    
    async def main_loop_logic(self, scores):
        # Sync and update weights logic
        self.metagraph.sync(subtensor=self.subtensor)
        bt.logging.info(f"🔄 Syncing metagraph with subtensor.")
        filtered_axons = [self.metagraph.axons[i] for i in self.get_filtered_axons()]
        responses = self.query_network(filtered_axons)
        self.process_responses(filtered_axons, responses, scores)




class TextToSpeechService(AIModelService):
    def __init__(self):
        super().__init__()  # Initializes base class components
        self.load_prompts()
        self.total_dendrites_per_query = 25  # Example value, adjust as needed
        self.minimum_dendrites_per_query = 3  # Example value, adjust as needed



    def load_prompts(self):
        if self.config.hub_key:
            gs_dev = load_dataset("speechcolab/gigaspeech", "dev", use_auth_token=self.config.hub_key)
            self.prompts = gs_dev['validation']['text']
        else:
            self.prompts = example_prompts  # Ensure example_prompts is defined globally

    async def run_async(self):
        step = 0

        while True:
            try:
                await self.main_loop_logic(step, self.scores)
                step += 1
                await asyncio.sleep(5)  # Adjust the sleep time as needed
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting TextToSpeechService.")
                break
            except Exception as e:
                print(f"An error occurred in TextToSpeechService: {e}")
                traceback.print_exc()

    async def main_loop_logic(self, step, scores):
        # Sync and update weights logic
        if step % 5 == 0:
            self.metagraph.sync(subtensor=self.subtensor)
            bt.logging.info(f"🔄 Syncing metagraph with subtensor.")

        if step % 2 == 0:
            filtered_axons = [self.metagraph.axons[i] for i in self.get_filtered_axons()]
            prompt = random.choice(self.prompts)
            responses = self.query_network(filtered_axons,prompt)
            self.process_responses(filtered_axons,responses, prompt, scores)

    def query_network(self,filtered_axons, prompt):
        # Network querying logic
        
        responses = self.dendrite.query(
            filtered_axons,
            lib.protocol.TextToSpeech(roles=["user"], text_input=prompt),
            deserialize=True,
            timeout=60,
        )
        return responses
    


    def process_responses(self,filtered_axons, responses, prompt, scores):
        for axon, response in zip(filtered_axons, responses):
            if response is not None and isinstance(response, lib.protocol.TextToSpeech):
                self.process_response(axon, response, prompt, scores)
        bt.logging.info(f"Updated Scores : {scores}")



    def process_response(self, axon, response, prompt, scores):
        # Process each response
        # Logic to convert the response to audio, save it, score it, and update weights
        speech_output = response.speech_output
        if speech_output is not None:
            self.handle_speech_output(axon, speech_output, prompt, scores, response.model_name)
            self.update_weights(scores)

    def handle_speech_output(self, axon, speech_output, prompt, scores, model_name):
        try:
            # Convert the list to a tensor
            speech_tensor = torch.Tensor(speech_output)

            # Normalize the speech data
            audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))

            # Convert to 32-bit PCM
            audio_data_int = (audio_data * 2147483647).type(torch.IntTensor)

            # Add an extra dimension to make it a 2D tensor
            audio_data_int = audio_data_int.unsqueeze(0)

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
            # Determine the sampling rate based on model name (if applicable)
            # For example, if model is 'suno/bark', set sampling rate to 24000
            sampling_rate = 24000 if model_name == "suno/bark" else 16000
            torchaudio.save(output_path, src=audio_data_int, sample_rate=sampling_rate)
            print(f"Saved audio file to {output_path}")

            # Score the output and update the weights
            score = self.score_output(output_path, prompt)
            self.update_score(axon, score, scores)

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

    def update_score(self, axon, new_score, scores):
        # Find the axon's index in the metagraph
        # axon_index = self.metagraph.uids.index(axon.uid)
        # alpha = self.config.alpha
        # scores[axon_index] = alpha * scores[axon_index] + (1 - alpha) * new_score


        uids = self.metagraph.uids.tolist()
        zipped_uids = list(zip(uids, self.metagraph.axons))
        uid_index = list(zip(*filter(lambda x: x[1] == axon, zipped_uids)))[0][0]
        alpha = self.config.alpha
        self.scores[uid_index] = alpha * self.scores[uid_index] + (1 - alpha) * new_score

        # Update the score for this axon
        # Assuming alpha is a weight factor for the moving average


        # Log the updated score
        bt.logging.info(f"Updated score for Hotkey {axon.hotkey}: {scores[uid_index]}")

    def get_filtered_axons(self):
        # If the metagraph has changed, update the weights.
        # Get the uids of all miners in the network.
        uids = self.metagraph.uids.tolist()
        # If there are more uids than scores, add more weights.
        if len(uids) > len(self.scores):
            bt.logging.trace("Adding more weights")
            size_difference = len(uids) - len(self.scores)
            new_scores = torch.zeros(size_difference, dtype=torch.float32)
            self.scores = torch.cat((self.scores, new_scores))
            del new_scores
        # If there are less uids than scores, remove some weights.
        queryable_uids = (self.metagraph.total_stake >= 0)
        
        # Remove the weights of miners that are not queryable.
        queryable_uids = queryable_uids * torch.Tensor([self.metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids])
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
        filtered_uids = list(zip(*filter(lambda x: x[1], zipped_uids)))[0]
        bt.logging.info(f"filtered_uids:{filtered_uids}")
        dendrites_to_query = random.sample( filtered_uids, min( dendrites_per_query, len(filtered_uids) ) )
        bt.logging.info(f"dendrites_to_query:{dendrites_to_query}")
        return dendrites_to_query

    def update_weights(self, scores):
        # Calculate new weights from scores
        weights = scores / torch.sum(scores)
        bt.logging.info(f"Setting weights: {weights}")

        # Process weights for the subnet
        processed_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor
        )
        bt.logging.info(f"Processed weights: {processed_weights}")
        bt.logging.info(f"Processed uids: {processed_uids}")

        # Set weights on the Bittensor network
        result = self.subtensor.set_weights(
            netuid=self.config.netuid,  # Subnet to set weights on
            wallet=self.wallet,         # Wallet to sign set weights using hotkey
            uids=processed_uids,        # Uids of the miners to set weights for
            weights=processed_weights   # Weights to set for the miners
        )

        if result:
            bt.logging.success('Successfully set weights.')
        else:
            bt.logging.error('Failed to set weights.')




def main():
    # Initialize the TextToSpeechService
    tts_service = TextToSpeechService()

    # Run the asynchronous loop of the TextToSpeechService
    asyncio.run(tts_service.run_async())

if __name__ == "__main__":
    main()
