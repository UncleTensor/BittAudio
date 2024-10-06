# Audio Generation Subnetwork Miner Guide
Welcome to the Miner's guide for the Audio Generation Subnetwork within the Bittensor network. This document provides instructions for setting up and running a Miner node in the network.

## Overview
Miners in the Audio Subnetwork are responsible for generating audio from text prompts received from Validators. Utilizing advanced text-to-music models, miners aim to produce high-fidelity, natural-sounding music. The quality of the generated audio directly influences the rewards miners receive.

## Installation
Follow these steps to install the necessary components:

**Set Conda Enviornment**
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
conda create -n {conda-env} python=3.10 -y
conda activate {conda-env}
```
**Install Repo**
```bash
git clone https://github.com/UncleTensor/BittAudio.git
cd BittAudio
pip install -e .
pip install -r requirements.txt
```
**Install pm2**
```bash
sudo apt install nodejs npm
sudo npm install pm2 -g
```

### Recommended GPU Configuration
- NVIDIA GeForce RTX A6000 GPUs are recommended for optimal performance.

### Running a Miner
 - To operate a miner, run the miner.py script with the necessary configuration.

### Miner Commands
```bash
pm2 start neurons/miner.py -- \
    --netuid 50 \
    --wallet.name {wallet_name} \
    --wallet.hotkey {hotkey_name} \
    --logging.trace \
    --music_path {ttm-model} \
    --axon.port {machine_port}
```

### Bittensor Miner Script Arguments:

| **Category**                   | **Argument**                         | **Default Value**          | **Description**                                                                                                       |
|---------------------------------|--------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Text To Music Model** | `--music_model`                           | 'facebook/musicgen-medium' ; 'facebook/musicgen-large'       | The model to use for Text-To-Music |
| **Music Finetuned Model** | `--music_path`                           | /path/to/model | The model to use for Text-To-Music |
| **Network UID** | `--netuid`                           |  Mainnet: 50        | The chain subnet UID. |
| **Bittensor Subtensor Arguments** | `--subtensor.chain_endpoint`        | -                          | Endpoint for Bittensor chain connection.|
|                                 | `--subtensor.network`                | -                          | Bittensor network endpoint.|
| **Bittensor Logging Arguments** | `--logging.debug`                    | -                          | Enable debugging logs.|
| **Bittensor Wallet Arguments**  | `--wallet.name`                      | -                          | Name of the wallet.|
|                                 | `--wallet.hotkey`                    | -                  | Hotkey path for the wallet.|
| **Bittensor Axon Arguments**    | `--axon.port`                        | -                          | Port number for the axon server.|
| **PM2 process name**    | `--pm2_name`                        | 'SN50Miner'                          | Name for the pm2 process for Auto Update. |





### License
Refer to the main README for the MIT License details.
