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

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

from classes.tts import TextToSpeechService 
from classes.vc import VoiceCloningService
from classes.ttm import MusicGenerationService

async def main():
    services = [
        TextToSpeechService(),
        MusicGenerationService(),
        VoiceCloningService(),
    ]

    # Initialize an empty list to hold our tasks
    tasks = []

    # Iterate through each service and create an asynchronous task for its run_async method
    for service in services:
        if isinstance(service, TextToSpeechService):
            service.new_wandb_run()  # Initialize the Weights & Biases run if the service is TextToSpeechService
        task = asyncio.create_task(service.run_async())
        tasks.append(task)

        await asyncio.sleep(0.1)  # Short delay between task initializations if needed

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
