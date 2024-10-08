# Audio Generation Subnetwork Validator Guide

Welcome to the Validator's guide for the Audio Generation Subnetwork within the Bittensor network. This document provides instructions for setting up and running a Validator node in the network.

## Overview
Validators initiate the audio generation process by providing prompts to the Miners and evaluate the quality of the generated audio. They play a crucial role in maintaining the quality standards of the network. The prompts will be generated with the help of the Corcel API, Product by Subnet 18, which provides a infinite range of prompts for Text-To-Music.

## Installation
Follow these steps to install the necessary components:

**Export Corcel API key**
```bash
echo "export CORCEL_API_KEY=XXXXXXXXXXXXXXX">>~/.bashrc && source ~/.bashrc
```

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
pip install audiocraft
git clone https://github.com/UncleTensor/BittAudio.git
cd BittAudio
pip install -e
pip install laion_clap==1.1.4
wandb login
```
**Install pm2**
```bash
sudo apt install nodejs npm
sudo npm install pm2 -g
```

## Running a Validator
- To operate a validator, run the validator.py script with the required command-line arguments.

## Validator Command
```bash
python neurons/validator.py 
```
```bash
pm2 start neurons/validator.py
```

change the default arguements from `lib/default_args.py`

### Bittensor Validator Script Arguments:

| **Category**                   | **Argument**                         | **Default Value**          | **Description**                                                                                                       |
|---------------------------------|--------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Configuration Arguments**     | `--alpha`                            | 0.9                        | The weight moving average scoring.                                                                                    |
|                                 | `--netuid`                           |  Mainnet: 50                          | The chain subnet UID.                                                                                                 |
| **Bittensor Subtensor Arguments** | `--subtensor.chain_endpoint`        | -                          | Endpoint for Bittensor chain connection.                                                                              |
|                                 | `--subtensor.network`                | -                          | Bittensor network endpoint.                                                                                          |
| **Bittensor Logging Arguments** | `--logging.debug`                    | -                          | Enable debugging logs.                                                                                               |
| **Bittensor Wallet Arguments**  | `--wallet.name`                      | -                          | Name of the wallet.                                                                                                  |
|                                 | `--wallet.hotkey`                    | -                  | Hotkey path for the wallet.                                                                                          |
| **PM2 process name**    | `--pm2_name`                        | 'SN50Miner'                          | Name for the pm2 process for Auto Update. |

### License
Refer to the main README for the MIT License details.

