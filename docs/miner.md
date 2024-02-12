# Audio Generation Subnetwork Miner Guide
Welcome to the Miner's guide for the Audio Generation Subnetwork within the Bittensor network. This document provides instructions for setting up and running a Miner node in the network.

## Overview
Miners in the Audio Subnetwork are responsible for generating audio from text prompts received from Validators. Utilizing advanced text-to-speech models, miners aim to produce high-fidelity, natural-sounding voice recordings. The quality of the generated audio directly influences the rewards miners receive.

## Installation
Follow these steps to install the necessary components:

```bash
git clone https://github.com/UncleTensor/AudioSubnet.git
cd AudioSubnet
git checkout main
pip install -e fseq/
pip install -r requirements.txt
python -m pip install -e . 
wandb login
```


### Recommended GPU Configuration
- NVIDIA GeForce RTX 3090 GPUs are recommended for optimal performance.

### Running a Miner
 - To operate a miner, run the miner.py script with the necessary configuration.

### Miner Commands
For running VC ElevenLabs API:
```bash
echo "export ELEVEN_API={your_api_key_here}">>~/.bashrc && source ~/.bashrc
```
Export your API key to environment variable

```bash
python neurons/miner.py \
    --netuid 16 \
    --wallet.name {wallet_name} \
    --wallet.hotkey {hotkey_name} \
    --logging.debug \
    --clone_model elevenlabs/eleven \
    --music_model facebook/musicgen-medium \
    --model elevenlabs/eleven \
    --axon.port {machine_port}
```

For running VC bark/voiceclone:
```bash
python neurons/miner.py \
    --netuid 16 \
    --wallet.name {wallet_name} \
    --wallet.hotkey {hotkey_name} \
    --logging.debug \
    --clone_model bark/voiceclone \
    --music_model facebook/musicgen-medium \
    --model {model} \
    --axon.port {machine_port}
```

### Bittensor Miner Script Arguments:

| **Category**                   | **Argument**                         | **Default Value**          | **Description**                                                                                                       |
|---------------------------------|--------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Text To Speech Model**    | `--model`                            | 'microsoft/speecht5_tts' ; 'elevenlabs/eleven' ; 'facebook/mms-tts-eng' ; 'suno/bark'   | The model to use for text-to-speech.|
| **Text To Music Model** | `--music_model`                           | 'facebook/musicgen-medium' ; 'facebook/musicgen-large'       | The model to use for Text-To-Music |
| **Voice Clone Model** | `--clone_model`                           | 'bark/voiceclone' ; 'elevenlabs/eleven'       | The model to use for Voice Clone |
| **Music Finetuned Model** | `--music_path`                           | /path/to/model | The model to use for Text-To-Music |
| **Voice Clone Finetuned Model** | `--bark_vc_path`                           | /path/to/model | The bark Finetuned model to use for Voice Clone |
| **Microsoft TTS Finetuned Model**    | `--ms_model_path`                        | /path/to/model | The Finetuned Microsoft tts model to be used for text-to-speech. |
| **Facebook TTS Finetuned Model**    | `--fb_model_path`                        | /path/to/model | The Finetuned Facebook tts model to be used for text-to-speech. |
| **Bark TTS Finetuned Model**    | `--bark_model_path`                        |  /path/to/model | The Finetuned Bark tts model to be used for text-to-speech. |
| **Network UID** | `--netuid`                           |  Mainnet: 16        | The chain subnet UID. |
| **Bittensor Subtensor Arguments** | `--subtensor.chain_endpoint`        | -                          | Endpoint for Bittensor chain connection.|
|                                 | `--subtensor.network`                | -                          | Bittensor network endpoint.|
| **Bittensor Logging Arguments** | `--logging.debug`                    | -                          | Enable debugging logs.|
| **Bittensor Wallet Arguments**  | `--wallet.name`                      | -                          | Name of the wallet.|
|                                 | `--wallet.hotkey`                    | -                  | Hotkey path for the wallet.|
| **Bittensor Axon Arguments**    | `--axon.port`                        | -                          | Port number for the axon server.|
| **Auto update repository**    | `--auto_update`                        | 'yes'                          | Auto update option for github repository updates. |





### License
Refer to the main README for the MIT License details.
