import os
import sys
import lib
import time
import torch
import typing
import argparse
import traceback
import torchaudio
import bittensor as bt
import ttm.protocol as protocol
# from ttm.protocol import MusicGeneration
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

class MusicGenerator:
    def __init__(self, model_path="facebook/musicgen-medium"):
        """Initializes the MusicGenerator with a specified model path."""
        self.model_name = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the processor and model
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def generate_music(self, prompt, token):
        """Generates music based on a given prompt and token count."""
        try:
            inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)
            audio_values = self.model.generate(**inputs, max_new_tokens=token)
            return audio_values[0, 0].cpu().numpy()
        except Exception as e:
            print(f"Error occurred with {self.model_name}: {e}")
            return None

# Configuration setup
def get_config():
    parser = argparse.ArgumentParser()

    # Add model selection for Text-to-Music
    parser.add_argument("--music_model", default='facebook/musicgen-medium', help="The model to be used for Music Generation.")
    parser.add_argument("--music_path", default=None, help="Path to a custom finetuned model for Music Generation.")

    # Add Bittensor specific arguments
    parser.add_argument("--netuid", type=int, default=50, help="The chain subnet uid.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)

    config = bt.config(parser)

    # Set up logging paths
    config.full_path = os.path.expanduser(f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/miner")

    # Ensure the logging directory exists
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    return config

# Main function
def main(config):
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running TTM miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")

    # Text-to-Music Model Setup
    try:
        if config.music_path:
            bt.logging.info(f"Using custom model for Text-To-Music from: {config.music_path}")
            ttm_models = MusicGenerator(model_path=config.music_path)
        elif config.music_model in ["facebook/musicgen-medium", "facebook/musicgen-large"]:
            bt.logging.info(f"Using Text-To-Music model: {config.music_model}")
            ttm_models = MusicGenerator(model_path=config.music_model)
        else:
            bt.logging.error(f"Invalid music model: {config.music_model}")
            exit(1)
    except Exception as e:
        bt.logging.error(f"Error initializing Text-To-Music model: {e}")
        exit(1)

    # Bittensor object setup
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error("Miner not registered. Run btcli register and try again.")
        exit()

    # Check the miner's subnet UID
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    ######################## Text to Music Processing ########################

    def music_blacklist_fn(synapse: protocol.MusicGeneration) -> typing.Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            bt.logging.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"
        elif synapse.dendrite.hotkey in metagraph.hotkeys and metagraph.S[metagraph.hotkeys.index(synapse.dendrite.hotkey)] < lib.MIN_STAKE:
            # Ignore requests from entities with low stake.
            bt.logging.trace(
                f"Blacklisting hotkey {synapse.dendrite.hotkey} with low stake"
            )
            return True, "Low stake"
        else:
            return False, "Accepted"

    # The priority function determines the request handling order.
    def music_priority_fn(synapse: protocol.MusicGeneration) -> float:
        caller_uid = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(metagraph.S[caller_uid])
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with stake: {priority}")
        return priority

    def convert_music_to_tensor(audio_file):
        """Convert the audio file to a tensor."""
        try:
            _, file_extension = os.path.splitext(audio_file)
            if file_extension.lower() in ['.wav', '.mp3']:
                audio, sample_rate = torchaudio.load(audio_file)
                return audio[0].tolist()  # Convert to tensor/list
            else:
                bt.logging.error(f"Unsupported file format: {file_extension}")
                return None
        except Exception as e:
            bt.logging.error(f"Error converting file: {e}")

    def ProcessMusic(synapse: protocol.MusicGeneration) -> protocol.MusicGeneration:
        bt.logging.info(f"Generating music with model: {config.music_path if config.music_path else config.music_model}")
        print(f"synapse.text_input: {synapse.text_input}")
        print(f"synapse.duration: {synapse.duration}")
        music = ttm_models.generate_music(synapse.text_input, synapse.duration)

        if music is None:
            bt.logging.error("No music generated!")
            return None
        try:
            sampling_rate = 32000
            write_wav("random_sample.wav", rate=sampling_rate, data=music)
            bt.logging.success("Music generated and saved to random_sample.wav")
            music_tensor = convert_music_to_tensor("random_sample.wav")
            synapse.music_output = music_tensor
            return synapse
        except Exception as e:
            bt.logging.error(f"Error processing music output: {e}")
            return None

    ######################## Attach Axon and Serve ########################

    axon = bt.axon(wallet=wallet, config=config)
    bt.logging.info(f"Axon {axon}")

    # Attach forward function for TTM processing
    axon.attach(
        forward_fn=ProcessMusic,
        blacklist_fn=music_blacklist_fn,
        priority_fn=music_priority_fn,
    )

    # Serve the axon on the network
    bt.logging.info(f"Serving axon on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}")
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # Start the miner's axon
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Keep the miner running
    bt.logging.info("Starting main loop")
    step = 0
    while True:
        try:
            # Periodically update knowledge of the network graph
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

        # Stop the miner safely
        except KeyboardInterrupt:
            axon.stop()
            break

        # Log any unexpected errors
        except Exception as e:
            bt.logging.error(f"unexpected error",traceback.format_exc())
            continue

# Entry point
if __name__ == "__main__":
    config = get_config()
    main(config)