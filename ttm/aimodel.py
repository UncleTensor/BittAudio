import bittensor as bt
import pandas as pd
import subprocess
import platform
import argparse
import inflect
import psutil
import GPUtil
import sys
import os
import re
from lib.default_args import default_args as args
from lib import __spec_version__ as spec_version


class AIModelService:
    _scores = None
    _base_initialized = False  # Class-level flag for one-time initialization
    version: int = spec_version  # Adjust version as necessary

    def __init__(self):
        self.config = self.get_config()
        self.sys_info = self.get_system_info()
        self.setup_paths()
        self.setup_logging()
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.p = inflect.engine()

        if not AIModelService._base_initialized:
            bt.logging.info(f"Wallet: {self.wallet}")
            bt.logging.info(f"Subtensor: {self.subtensor}")
            bt.logging.info(f"Dendrite: {self.dendrite}")
            bt.logging.info(f"Metagraph: {self.metagraph}")
            AIModelService._base_initialized = True

        if AIModelService._scores is None:
            AIModelService._scores = self.metagraph.E.copy()
        self.scores = AIModelService._scores
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

    def get_config(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--alpha", default=0.75, type=float, help="The weight moving average scoring.")
        parser.add_argument("--custom", default="my_custom_value", help="Adds a custom value to the parser.")
        parser.add_argument("--subtensor.network", type=str, default=args['subtensor_network'], help="The logging directory.")
        parser.add_argument("--netuid", default=50, type=int, help="The chain subnet uid.")
        parser.add_argument("--wallet.name", type=str, default=args['wallet_name'], help="The wallet name.")
        parser.add_argument("--wallet.hotkey", type=str, default=args['wallet_hotkey'], help="The wallet hotkey.")

        # Add Bittensor specific arguments
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)

        # Parse and return the config
        config = bt.config(parser)
        return config

    def priority_uids(self, metagraph):
        hotkeys = metagraph.hotkeys  # List of hotkeys
        coldkeys = metagraph.coldkeys  # List of coldkeys
        UIDs = range(len(hotkeys))  # Assuming UID is the index of neurons
        stakes = metagraph.S.numpy()  # Total stake
        emissions = metagraph.E.numpy()  # Emission

        # Create a DataFrame from the metagraph data
        df = pd.DataFrame({
            "UID": UIDs,
            "HOTKEY": hotkeys,
            "COLDKEY": coldkeys,
            "STAKE": stakes,
            "EMISSION": emissions,
            "AXON": metagraph.axons,
        })

        # Filter and sort the DataFrame
        df = df[df['STAKE'] < 500]
        df = df.sort_values(by=["EMISSION"], ascending=False)
        uid = df.iloc[0]['UID']
        axon_info = df.iloc[0]['AXON']

        result = [(uid, axon_info)]
        return result

    def get_system_info(self):
        system_info = {
            "OS Version": platform.platform(),
            "CPU Count": os.cpu_count(),
            "RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB", 
        }

        gpus = GPUtil.getGPUs()
        if gpus:
            system_info["GPU"] = gpus[0].name 

        # Convert dictionary to list of strings for logging purposes
        tags = [f"{key}: {value}" for key, value in system_info.items()]
        return tags

    def setup_paths(self):
        # Set the project root path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Add project root to sys.path
        sys.path.insert(0, project_root)

    def convert_numeric_values(self, input_prompt):
        # Regular expression to identify date patterns
        date_pattern = r'(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)|(\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b)'

        # Regular expression to find ordinal numbers
        ordinal_pattern = r'\b\d{1,2}(st|nd|rd|th)\b'

        # Regular expression to find numeric values with and without commas, excluding those part of date patterns and ordinals
        numeric_pattern = r'\b(?<![\d,/-])(\d{1,3}(?:,\d{3})*|\d+)(?![,\d/-])(?!\d*(st|nd|rd|th))\b'

        # Protect date formats from changes
        date_matches = re.findall(date_pattern, input_prompt)
        date_replacements = {}
        for i, date in enumerate(date_matches):
            placeholder = f'{{DATE{i}}}'
            input_prompt = input_prompt.replace(date[0], placeholder)
            date_replacements[placeholder] = date[0]

        # Protect ordinal numbers from changes
        ordinal_matches = re.findall(ordinal_pattern, input_prompt)
        ordinal_replacements = {}
        for i, ordinal in enumerate(ordinal_matches):
            placeholder = f'{{ORDINAL{i}}}'
            input_prompt = input_prompt.replace(ordinal, placeholder)
            ordinal_replacements[placeholder] = ordinal

        # Convert and replace numeric values
        numeric_matches = re.findall(numeric_pattern, input_prompt)
        for match in numeric_matches:
            matched_str = match if isinstance(match, str) else match[0]  # Ensure match is treated as string
            numeric_value = int(matched_str.replace(",", ""))  # Remove commas and convert to integer
            numeric_in_words = self.p.number_to_words(numeric_value)
            input_prompt = input_prompt.replace(matched_str, numeric_in_words, 1)  # Replace only the first occurrence

        # Reinsert dates into the input prompt
        for placeholder, date_str in date_replacements.items():
            input_prompt = input_prompt.replace(placeholder, date_str)

        # Convert and reinsert ordinal numbers into the input prompt
        for placeholder, ordinal_str in ordinal_replacements.items():
            numeric_parts = re.findall(r'\d+', ordinal_str)
            if numeric_parts:  # Check if there are numeric parts found
                numeric_part = int(numeric_parts[0])  # Convert the first found numeric part to integer
                ordinal_in_words = self.p.number_to_words(numeric_part, ordinal=True)  # Convert to ordinal words
                input_prompt = input_prompt.replace(placeholder, ordinal_in_words)
            else:
                # Revert back to the original text if no numeric part is found
                input_prompt = input_prompt.replace(placeholder, ordinal_str)

        return input_prompt

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

    def update_score(self, axon, new_score, service):
        try:
            uids = self.metagraph.uids.tolist()
            zipped_uids = list(zip(uids, self.metagraph.axons))
            uid_index = next(index for index, ax in zipped_uids if ax == axon)

            alpha = self.config.alpha
            self.scores[uid_index] = alpha * self.scores[uid_index] + (1 - alpha) * new_score
            bt.logging.info(f"Updated score for {service} Hotkey {axon.hotkey}: {self.scores[uid_index]}")
        except Exception as e:
            print(f"Error updating the score: {e}")

    def punish(self, axon, service, punish_message):
        '''Punish the axon for returning an invalid response'''
        try:
            uids = self.metagraph.uids.tolist()
            zipped_uids = list(zip(uids, self.metagraph.axons))
            uid_index = next(index for index, ax in zipped_uids if ax == axon)

            alpha = self.config.alpha
            self.scores[uid_index] = alpha * self.scores[uid_index] + (1 - alpha) * (-0.1)
            self.scores[uid_index] = max(self.scores[uid_index], 0)  # Ensure scores don't go below 0
            bt.logging.info(f"Punished Hotkey {axon.hotkey} using {service}: {self.scores[uid_index]}")
        except Exception as e:
            print(f"Error punishing the axon: {e}")

    def get_git_commit_hash(self):
        try:
            # Get the current Git commit hash
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            return commit_hash
        except subprocess.CalledProcessError:
            bt.logging.error("Failed to get git commit hash. '.git' folder is missing")
            return None

    async def run_async(self):
        raise NotImplementedError
