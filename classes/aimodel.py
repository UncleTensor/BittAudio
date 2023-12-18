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