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
import lib.utils
import lib
import traceback
from classes.tts import TextToSpeechService,AIModelService


def main():
    AIModelService()
    # Initialize the TextToSpeechService
    tts_service = TextToSpeechService()
    # Run the asynchronous loop of the TextToSpeechService
    asyncio.run(tts_service.run_async())

if __name__ == "__main__":
    main()
