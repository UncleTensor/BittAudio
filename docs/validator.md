# Audio Generation Subnetwork Validator Guide

Welcome to the Validator's guide for the Audio Generation Subnetwork within the Bittensor network. This document provides instructions for setting up and running a Validator node in the network.

## Overview
Validators initiate the audio generation process by providing prompts to the Miners and evaluate the quality of the generated audio. They play a crucial role in maintaining the quality standards of the network. The prompts will be generated with the help of the Corcel API, Product by Subnet 18, which provides a infinite range of prompts for Text-To-Music.

## Installation
Follow these steps to install the necessary components:

**Set Conda Enviornment**
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
bash
~/miniconda3/bin/conda init zsh
conda create -n {conda-env} python=3.10 -y
conda activate {conda-env}
```
**Install Repo**
```bash
sudo apt update
sudo apt install build-essential -y
git clone https://github.com/IamHussain503/NewMusic.git
cd NewMusic
pip install -e.
pip install audiocraft
pip install laion_clap==1.1.4
pip install git+https://github.com/haoheliu/audioldm_eval
pip install git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt
sudo mkdir -p /tmp/music
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

## Miners logs in Validator

If it is required to check miners logs in the validator, one can go ahead to ~/.pm2/logs directory and grep the miners scoring logs
as follows:

sudo grep -a -A 10 "Raw score for hotkey:5DXTGaAQm99AEAvhMRqWQ77b1aob4mAXwX" ~/.pm2/logs/validator-out.log

sudo grep -a -A 10 "Normalized score for hotkey:5DXTGaAQm99AEAvhMRqWQ77b1aob4mAXwX" ~/.pm2/logs/validator-out.log

### License
Refer to the main README for the MIT License details.

