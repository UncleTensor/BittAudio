# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

# Bittensor Validator Template:
# TODO(developer): Rewrite based on protocol defintion.

# Step 1: Import necessary libraries and modules
import os
import time
import torch
import argparse
import traceback
import bittensor as bt
from scipy.io import wavfile
import asyncio
from datasets import load_dataset
import random
import csv
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import torchaudio

# Adjust the path to include the directory where 'template' is located
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path
ttv = os.path.abspath(os.path.join(current_script_dir, "ttvMain"))

# Check if the path is already in sys.path
if ttv not in sys.path:
    # Insert it at the beginning of the sys.path list
    sys.path.insert(0, ttv)
# import this repo
import template
# Set the sampling rate (you can adjust this as needed)
sampling_rate = 16000  # 16 kHz is a common choice

# Define a threshold value (you can adjust this as needed)
threshold = 0.8


example_prompts = [
    "Please read me a bedtime story.",
    "Translate the following sentence into French: 'Hello, how are you?'",
    "Generate an audio file for the poem 'The Road Not Taken' by Robert Frost.",
    "Convert the following news article into speech: [Paste the article here]."
    "Read the weather forecast for tomorrow.",
    "Narrate a description of your favorite book.",
    "Create an audio version of the user manual for a smartphone.",
    "Generate speech for a conversation between two characters in a novel.",
    "Read aloud the recipe for chocolate chip cookies.",
    "Translate 'Thank you' into multiple languages and speak them.",
    "Summarize the latest science news article in under 3 minutes.",
    "Generate an audiobook for the first chapter of 'Pride and Prejudice.'",
    "Read me a famous speech, like Martin Luther King Jr.'s 'I Have a Dream.'",
    "Convert a Wikipedia article about space exploration into an audio.",
    "Speak the lyrics of a popular song.",
    "Narrate a travelogue for a trip to a tropical island.",
    "Create an audio guide for a historical monument or landmark.",
    "Read a humorous short story.",
    "Translate and narrate a short phrase in Morse code.",
    "Generate speech for a fictional character's monologue."
]

# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
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
    # Add argument for the threshold
    parser.add_argument(
        "--threshold", default=0.68, type=float, help="The threshold for response scoring."
    )
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
    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")
    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0
    d = bt.dendrite()
    while True:
        try:
            # TODO(developer): Define how the validator selects a miner to query, how often, etc.
            # Broadcast a query to all miners on the network.
            random_prompt = random.choice(prompts)
            responses = dendrite.query(
                metagraph.axons,
                template.protocol.TextToSpeech(roles=["user"], text_input=random_prompt),
                deserialize=True,
                timeout=60,
            )

            # TODO(developer): Define how the validator scores responses.
            # Adjust the scores based on responses from miners.
            for i, resp_i in enumerate(responses[:-1]): # The last response is the dendrite's own response - x3r!
                if isinstance(resp_i, template.protocol.TextToSpeech):
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
                            output_path = os.path.join('/tmp', f'output_{metagraph.axons[i].hotkey}.wav')
                            # set model sampling rate to 24000 if the model is Suno Bark
                            if resp_i.model_name == "suno/bark":
                                torchaudio.save(output_path, src=audio_data_int, sample_rate=24000)
                            else:
                                torchaudio.save(output_path, src=audio_data_int, sample_rate=16000)
                            # wavfile.write(output_path, sampling_rate, audio_tensor)
                            score = template.reward.score(output_path, text_input)
                            # Get the current time
                            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                            # Append the score, time, and filename to the CSV file
                            with open('scores.csv', 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([output_path, score, current_time])

                            # print the csv file
                            df = pd.read_csv('scores.csv')
                            print(tabulate(df, ["No #", "Files w/ Hotkey", "Score", "Time"], tablefmt='psql'))


                            # Update the global score of the miner.
                            # This score contributes to the miner's weight in the network.
                            # A higher weight means that the miner has been consistently responding correctly.
                            scores[i] = config.alpha * scores[i] + (1 - config.alpha) * score
                            
                        except Exception as e:
                            bt.logging.error(f"Error writing WAV file: {e}")
                    else:
                        bt.logging.warning(f"Received None speech_output for prompt: {text_input}. Skipping.")

            bt.logging.info(f"Scores: {scores}")
            # Periodically update the weights on the Bittensor blockchain.
            if (step + 1) % 2 == 0:
                # TODO(developer): Define how the validator normalizes scores before setting weights.
                weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
                bt.logging.info(f"Setting weights: {weights}")
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result = subtensor.set_weights(
                    netuid=config.netuid,  # Subnet to set weights on.
                    wallet=wallet,  # Wallet to sign set weights using hotkey.
                    uids=metagraph.uids,  # Uids of the miners to set weights for.
                    weights=weights,  # Weights to set for the miners.
                    wait_for_inclusion=True,
                )
                if result:
                    bt.logging.success("Successfully set weights.")
                else:
                    bt.logging.error("Failed to set weights.")

            # End the current step and prepare for the next iteration.
            step += 1
            # Resync our local state with the latest state from the blockchain.
            metagraph = subtensor.metagraph(config.netuid)
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            time.sleep(bt.__blocktime__)

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            bt.logging.error(e)
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
