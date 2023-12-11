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
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha", default=0.9, type=float, help="The weight moving average scoring."
    )
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument(
        "--custom", default="my_custom_value", help="Adds a custom value to the parser."
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    # Supply Huggingface hub API key --hub_key default is None
    parser.add_argument("--hub_key", type=str, default=None, help="Supply the Huggingface Hub API key for prompt dataset")
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/validator.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "validator",
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config


def main(config):
    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    if config.hub_key:
        # Load the dataset from Hugging Face using the API key
        gs_dev = load_dataset("speechcolab/gigaspeech", "dev", use_auth_token=config.hub_key)
        prompts = gs_dev['validation']['text']
    else:
        # Use the example prompts if no API key is provided
        prompts = example_prompts

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite(wallet=wallet)
    bt.logging.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other validators and miners.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again."
        )
        exit()

    # Each validator gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 6: Set up initial scoring weights for validation
    bt.logging.info("Building validation weights.")
    scores = torch.zeros_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")
    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0

    curr_block = subtensor.block
    total_dendrites_per_query = 25 
    minimum_dendrites_per_query = 3 
    last_updated_block = curr_block - (curr_block % 100)
    last_reset_weights_block = curr_block
    current_block = subtensor.block


    while True:
        try:
            # Per 10 blocks, sync the subtensor state with the blockchain.
            if step % 5 == 0:
                metagraph.sync(subtensor = subtensor)
                bt.logging.info(f"🔄 Syncing metagraph with subtensor.")

            # If the metagraph has changed, update the weights.
            # Get the uids of all miners in the network.
            uids = metagraph.uids.tolist()
            # If there are more uids than scores, add more weights.
            if len(uids) > len(scores):
                bt.logging.trace("Adding more weights")
                size_difference = len(uids) - len(scores)
                new_scores = torch.zeros(size_difference, dtype=torch.float32)
                scores = torch.cat((scores, new_scores))
                del new_scores
            # If there are less uids than scores, remove some weights.
            queryable_uids = (metagraph.total_stake >= 0)
            bt.logging.info(f"queryable_uids:{queryable_uids}")
            
            # Remove the weights of miners that are not queryable.
            queryable_uids = queryable_uids * torch.Tensor([metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in uids])
            active_miners = torch.sum(queryable_uids)
            dendrites_per_query = total_dendrites_per_query

            # if there are no active miners, set active_miners to 1
            if active_miners == 0:
                active_miners = 1
            # if there are less than dendrites_per_query * 3 active miners, set dendrites_per_query to active_miners / 3
            if active_miners < total_dendrites_per_query * 3:
                dendrites_per_query = int(active_miners / 3)
            else:
                dendrites_per_query = total_dendrites_per_query
            
            # less than 3 set to 3
            if dendrites_per_query < minimum_dendrites_per_query:
                    dendrites_per_query = minimum_dendrites_per_query
            # zip uids and queryable_uids, filter only the uids that are queryable, unzip, and get the uids
            zipped_uids = list(zip(uids, queryable_uids))
            filtered_uids = list(zip(*filter(lambda x: x[1], zipped_uids)))[0]
            bt.logging.info(f"filtered_uids:{filtered_uids}")
            dendrites_to_query = random.sample( filtered_uids, min( dendrites_per_query, len(filtered_uids) ) )
            bt.logging.info(f"dendrites_to_query:{dendrites_to_query}")
                
                        
            # every 2 minutes, query the miners
            try:
                # Filter metagraph.axons by indices saved in dendrites_to_query list
                filtered_axons = [metagraph.axons[i] for i in dendrites_to_query]
                bt.logging.info(f"filtered_axons: {filtered_axons}")
                if step % 2 == 0: # 
                    bt.logging.info(f"Querying dendrites: {filtered_axons}")
                    # Broadcast a GET_DATA query to filtered miners on the network.
                    random_prompt = random.choice(prompts)
                    responses = dendrite.query(
                        filtered_axons,
                        lib.protocol.TextToSpeech(roles=["user"], text_input=random_prompt),
                        deserialize=True,
                        timeout=60,
                    )


                    # Adjust the scores based on responses from miners.
                    for iax, resp_i in zip(filtered_axons, responses): 
                        if isinstance(resp_i, lib.protocol.TextToSpeech):
                            #The response has been deserialized into the expected class
                            # Now you can access its properties
                            text_input = resp_i.text_input

                            # Check if speech_output is not None before processing
                            speech_output = resp_i.speech_output
                            if speech_output is not None:
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
                                    output_path = os.path.join('/tmp', f'output_{iax.hotkey}.wav')
                                    # Check if any WAV file with .wav extension exists and delete it
                                    existing_wav_files = [f for f in os.listdir('/tmp') if f.endswith('.wav')]
                                    for existing_file in existing_wav_files:
                                        try:
                                            os.remove(os.path.join('/tmp', existing_file))
                                        except Exception as e:
                                            bt.logging.error(f"Error deleting existing WAV file: {e}")
                                    # set model sampling rate to 24000 if the model is Suno Bark
                                    if resp_i.model_name == "suno/bark":
                                        torchaudio.save(output_path, src=audio_data_int, sample_rate=24000)
                                        print(f"Saved audio file to suno/bark -----{output_path}")
                                    else:
                                        torchaudio.save(output_path, src=audio_data_int, sample_rate=16000)
                                        print(f"Saved audio file to {output_path}")
                                    # wavfile.write(output_path, sampling_rate, audio_tensor)
                                    score = lib.reward.score(output_path, text_input)
                                    bt.logging.info(f"Score after saving the file -------------- : {score}")
                                    # Get the current time
                                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                                    # Append the score, time, and filename to the CSV file
                                    with open('scores.csv', 'a', newline='') as csvfile:
                                        writer = csv.writer(csvfile)
                                        writer.writerow([output_path, score, current_time])

                                    # print the csv file
                                    df = pd.read_csv('scores.csv')
                                    # make columns headers of df
                                    df.columns = ["Files w/ Hotkey", "Score", "Time"]
                                    # print the row if it is not empty
                                    if not df.empty:
                                        print(df.tail(1))
                                    # Delete the WAV file
                                    os.remove(output_path)

                                    # Update the global score of the miner.
                                    # This score contributes to the miner's weight in the network.
                                    # A higher weight means that the miner has been consistently responding correctly.
                                    zipped_uids = list(zip(uids, metagraph.axons))
                                    uid_index = list(zip(*filter(lambda x: x[1] == iax, zipped_uids)))[0][0]
                                    scores[uid_index] = config.alpha * scores[uid_index] + (1 - config.alpha) * score
                                    
                                except Exception as e:
                                    bt.logging.error(f"Error writing WAV file: {e}")
                            else:
                                bt.logging.warning(f"Received None speech_output for prompt: {text_input}. Skipping.")

                    bt.logging.info(f"Scores: {scores}")
                    
                    current_block = subtensor.block
                    if current_block - last_updated_block > 100:
                        
                        weights = scores / torch.sum(scores)
                        bt.logging.info(f"Setting weights: {weights}")
                        # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                        (
                            processed_uids,
                            processed_weights,
                        ) = bt.utils.weight_utils.process_weights_for_netuid(
                            uids=metagraph.uids,
                            weights=weights,
                            netuid=config.netuid,
                            subtensor=subtensor
                        )
                        bt.logging.info(f"Processed weights: {processed_weights}")
                        bt.logging.info(f"Processed uids: {processed_uids}")
                        result = subtensor.set_weights(
                            netuid = config.netuid, # Subnet to set weights on.
                            wallet = wallet, # Wallet to sign set weights using hotkey.
                            uids = processed_uids, # Uids of the miners to set weights for.
                            weights = processed_weights, # Weights to set for the miners.
                        )
                        last_updated_block = current_block
                        if result: bt.logging.success('Successfully set weights.')
                        else: bt.logging.error('Failed to set weights.')

                # End the current step and prepare for the next iteration.
                step += 1

                if last_reset_weights_block + 1800 < current_block:
                    bt.logging.trace(f"Clearing weights for validators and nodes without IPs")
                    last_reset_weights_block = current_block

                
                    # set all nodes without ips set to 0
                    scores = scores * torch.Tensor([metagraph.neurons[uid].axon_info.ip != '0.0.0.0' for uid in metagraph.uids])
                    
                # Resync our local state with the latest state from the blockchain.
                metagraph = subtensor.metagraph(config.netuid)
                # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
                time.sleep(bt.__blocktime__  * 5)

            except Exception as e:
                bt.logging.error(f"Error querying or processing responses: {e}")
                traceback.print_exc()


        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    config = get_config()
    # Run the main function.
    main(config)
