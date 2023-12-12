# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
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

# Bittensor Miner lib:
# TODO(developer): Rewrite based on protocol and validator defintion.

# Step 1: Import necessary libraries and modules
from scipy.io.wavfile import write as write_wav
import bittensor as bt
import numpy as np
import torchaudio
import traceback
import argparse
import typing
import torch
import wave
import time
import sys
import os

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
print("Contents of 'audiosubnet':", os.listdir(audio_subnet_path))
# import this repo
from models.text_to_speech_models import TextToSpeechModels
from models.text_to_speech_models import SunoBark
from models.text_to_speech_models import EnglishTextToSpeech
import lib.utils
import lib



def get_config():
    # Step 2: Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom miner arguments to the parser.
    parser.add_argument(
        "--custom", default="my_custom_value", help="Adds a custom value to the parser."
    )
    parser.add_argument(
        "--model", default='microsoft/speecht5_tts', help="The model to use for text-to-speech." # suno/bark-small
    )
    parser.add_argument("--auto_update", default="yes", help="Auto update")
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 lib/miner.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


# Main takes the config and starts the miner.
def main(config):
    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

        # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Check the supplied model and log the appropriate information.
    if config.model == "microsoft/speecht5_tts":
        bt.logging.info("Using the TextToSpeechModels with the supplied model: microsoft/speecht5_tts")
        tts_models = TextToSpeechModels()
    elif config.model == "facebook/mms-tts-eng":
        bt.logging.info("Using the English Text-to-Speech with the supplied model: facebook/mms-tts-eng")
        tts_models = EnglishTextToSpeech()
    elif config.model == "suno/bark":
        bt.logging.info("Using the SunoBark with the supplied model: suno/bark")
        tts_models = SunoBark()
    elif config.model is None:
        bt.logging.error("Model name was not supplied. Exiting the program.")
        exit(1)
    else:
        bt.logging.error(f"Wrong model was supplied: {config.model}. Exiting the program.")
        exit(1)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour miner: {wallet} is not registered to chain connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # Step 5: Set up miner functionalities
    # The following functions control the miner's response to incoming requests.
    # The blacklist function decides if a request should be ignored.
    def speech_blacklist_fn(synapse: lib.protocol.TextToSpeech) -> typing.Tuple[bool, str]:
        # TODO(developer): Define how miners should blacklist requests. This Function
        # Runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        # The synapse is instead contructed via the headers of the request. It is important to blacklist
        # requests before they are deserialized to avoid wasting resources on requests that will be ignored.
        # Below: Check that the hotkey is a registered entity in the metagraph.
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        # TODO(developer): In practice it would be wise to blacklist requests from entities that
        # are not validators, or do not have enough stake. This can be checked via metagraph.S
        # and metagraph.validator_permit. You can always attain the uid of the sender via a
        # metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.
        # Otherwise, allow the request to be processed further.
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    def speech_priority_fn(synapse: lib.protocol.TextToSpeech) -> float:
        # TODO(developer): Define how miners should prioritize requests.
        # Miners may recieve messages from multiple entities at once. This function
        # determines which request should be processed first. Higher values indicate
        # that the request should be processed first. Lower values indicate that the
        # request should be processed later.
        # Below: simple logic, prioritize requests from entities with more stake.
        caller_uid = metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def ProcessSpeech(synapse: lib.protocol.TextToSpeech) -> lib.protocol.TextToSpeech:
        bt.logging.debug("The prompt recieved from validator!")
        # Here we use the models class to generate the speech
        speech = tts_models.generate_speech(synapse.text_input)
        if config.model == "facebook/mms-tts-eng":
            # Assuming 'output' is a PyTorch tensor.
            # Normalize your data to -1 to 1 if not already
            audio_data = speech / torch.max(torch.abs(speech))

            # # If the audio is mono, ensure it has a channel dimension
            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0)

            # convert to 32-bit PCM
            audio_data_int = (audio_data * 2147483647).type(torch.IntTensor)

            # Save the audio data as integers
            torchaudio.save('speech.wav', src=audio_data_int, sample_rate=16000)
            # Open the WAV file and read the frames
            sample_width = None
            try:
                with wave.open('speech.wav', 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    sample_width = wav_file.getsampwidth()
            except Exception as e:
                print(f"An error occurred while reading the audio data: {e}")
            # Initialize dtype to a default value
            dtype = None
            # Determine the correct dtype based on sample width
            # Commonly, sample width is 2 bytes for 16-bit audio
            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 1:
                dtype = np.int8
            elif sample_width == 4:
                dtype = np.int32

            # Check if dtype has been assigned a value
            if dtype is None:
                print(f"Unexpected sample width: {sample_width}")
                return

            # Convert the bytes data to a numpy array
            audio_array = np.frombuffer(frames, dtype=dtype)
            # Convert the numpy array to a list
            speech = audio_array.tolist()

        # Check if 'speech' contains valid audio data
        if speech is None:
            bt.logging.error("No speech generated!")
            return None
        else:
            try:
                print("Speech generated!")
                if config.model == "facebook/mms-tts-eng":
                    # Convert the list to a tensor
                    speech_tensor = torch.Tensor(speech)

                    # Normalize the speech data
                    audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))

                    # Convert to 32-bit PCM
                    audio_data_int = (audio_data * 2147483647).type(torch.IntTensor)

                    # Add an extra dimension to make it a 2D tensor
                    audio_data_int = audio_data_int.unsqueeze(0)

                    # Save the audio data as a .wav file
                    # torchaudio.save('speech_output.wav', src=audio_data_int, sample_rate=16000)
                    synapse.speech_output = speech  # Convert PyTorch tensor to a list

                elif config.model == "suno/bark":
                    # Convert the list to a tensor
                    # Move the audio array back to CPU for saving to disk
                    speech = speech.cpu().numpy().squeeze()
                    # write_wav("output_audio.wav", 24000, speech)
                    synapse.model_name = config.model
                    synapse.speech_output = speech.tolist()
                else:
                    synapse.speech_output = speech.tolist()  # Convert PyTorch tensor to a list
                return synapse
            except Exception as e:
                print(f"An error occurred: {e}")

    # Step 6: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon(wallet=wallet, config=config)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn=ProcessSpeech,
        blacklist_fn=speech_blacklist_fn,
        priority_fn=speech_priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving axon {ProcessSpeech} on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
    )
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 7: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # TODO(developer): Define any additional operations to be performed by the miner.
            # Below: Periodically update our knowledge of the network graph.
            if step % 500 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[my_subnet_uid]:.6f} | "
                    f"Rank:{metagraph.R[my_subnet_uid]:.6f} | "
                    f"Trust:{metagraph.T[my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[my_subnet_uid]:.6f} | "
                    f"Incentive:{metagraph.I[my_subnet_uid]:.6f} | "
                    f"Emission:{metagraph.E[my_subnet_uid]}"
                )
                bt.logging.info(log)
            step += 1
            time.sleep(1)

            if step % 1000 == 0 and config.auto_update == "yes":
                lib.utils.update_repo()

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
# Entry point for the script
if __name__ == "__main__":
    config = get_config()
    main(config)
