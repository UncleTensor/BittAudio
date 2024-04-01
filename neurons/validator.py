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

import os
import sys
import asyncio
import datetime as dt
import wandb
import bittensor as bt
# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

from lib.globals import service_flags
from classes.tts import TextToSpeechService 
from classes.vc import VoiceCloningService
from classes.ttm import MusicGenerationService
from classes.aimodel import AIModelService



class AIModelController():
    def __init__(self):
        self.aimodel = AIModelService()
        self.text_to_speech_service = TextToSpeechService()
        self.music_generation_service = MusicGenerationService()
        self.voice_cloning_service = VoiceCloningService()
        self.current_service = self.text_to_speech_service
        self.service = service_flags
        self.last_run_start_time = dt.datetime.now()

    async def run_services(self):
        while True:
            self.check_and_update_wandb_run()
            if isinstance(self.current_service, TextToSpeechService) and self.service["TextToSpeechService"]:
                await self.current_service.run_async()
                self.current_service = self.music_generation_service
            elif isinstance(self.current_service, MusicGenerationService) and self.service["MusicGenerationService"]:
                await self.current_service.run_async()
                self.current_service = self.voice_cloning_service
            elif isinstance(self.current_service, VoiceCloningService) and self.service["VoiceCloningService"]:
                await self.current_service.run_async()
                self.current_service = self.text_to_speech_service

    def check_and_update_wandb_run(self):
        # Calculate the time difference between now and the last run start time
        current_time = dt.datetime.now()
        time_diff = current_time - self.last_run_start_time
        # Check if 4 hours have passed since the last run start time
        if time_diff.total_seconds() >= 4 * 3600 :  # 4 hours * 3600 seconds/hour
            self.last_run_start_time = current_time  # Update the last run start time to now
            if self.wandb_run:
                wandb.finish()  # End the current run
            self.new_wandb_run()  # Start a new run

    def new_wandb_run(self):
        now = dt.datetime.now()
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = f"Validator-{self.aimodel.uid}-{run_id}"
        commit = self.aimodel.get_git_commit_hash()
        self.wandb_run = wandb.init(
            name=name,
            project="AudioSubnet_Valid",
            entity="subnet16team",
            config={
                "uid": self.aimodel.uid,
                "hotkey": self.aimodel.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "Validator",
                "tao (stake)": self.aimodel.metagraph.neurons[self.aimodel.uid].stake.tao,
                "commit": commit,
            },
            tags=self.aimodel.sys_info,
            allow_val_change=True,
            anonymous="allow",
        )
        bt.logging.debug(f"Started a new wandb run: {name}")

async def main():
    controller = AIModelController()
    controller.new_wandb_run()
    await controller.run_services()

if __name__ == "__main__":
    asyncio.run(main())