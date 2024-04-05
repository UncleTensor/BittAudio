# Audio Generation Subnetwork Validator Guide

Welcome to the Validator's guide for the Audio Generation Subnetwork within the Bittensor network. This document provides instructions for setting up and running a Validator node in the network.

## Overview
Validators initiate the audio generation process by providing prompts to the Miners and evaluate the quality of the generated audio. They play a crucial role in maintaining the quality standards of the network. The prompts will be generated with the help of the Corcel API, Product by Subnet 18, which provides a infinite range of prompts for Text-To-Speech, Voice Clone and Text-To-Music.

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
git clone https://github.com/UncleTensor/AudioSubnet.git
cd AudioSubnet
pip install -e fseq/
pip install -e .
wandb login
```
**Install pm2**
```bash
sudo apt install nodejs npm
sudo npm install pm2 -g
```

## Running a Validator
- To operate a validator, run the validator.py script with the required command-line arguments.

## Starting with Auto Update script
 - To start the validator with auto-update, run the start_valid.py script with the required command-line arguments.
 - The script will automatically update the validator with the latest version.
 - [Note: the validator will have to run this auto-update script in a screen session]

## Validator Command
**With Auto Update ON**
```bash
python scripts/start_valid.py \
    --pm2_name {name} \
    --netuid 16 \
    --wallet.name {wallet_name} \
    --wallet.hotkey {hotkey_name} \
    --logging.debug \
```

**With Auto Update OFF**
```bash
python neurons/validator.py \
    --netuid 16 \
    --wallet.name {wallet_name} \
    --wallet.hotkey {hotkey_name} \
    --logging.debug \
```
```bash
pm2 start neurons/validator.py -- \
    --name {name} \
    --interpreter python3 \
    --netuid 16 \
    --wallet.name {wallet_name} \
    --wallet.hotkey {hotkey_name} \
    --logging.debug
```

### Bittensor Validator Script Arguments:

| **Category**                   | **Argument**                         | **Default Value**          | **Description**                                                                                                       |
|---------------------------------|--------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Configuration Arguments**     | `--alpha`                            | 0.9                        | The weight moving average scoring.                                                                                    |
|                                 | `--netuid`                           |  Mainnet: 16                          | The chain subnet UID.                                                                                                 |
| **Bittensor Subtensor Arguments** | `--subtensor.chain_endpoint`        | -                          | Endpoint for Bittensor chain connection.                                                                              |
|                                 | `--subtensor.network`                | -                          | Bittensor network endpoint.                                                                                          |
| **Bittensor Logging Arguments** | `--logging.debug`                    | -                          | Enable debugging logs.                                                                                               |
| **Bittensor Wallet Arguments**  | `--wallet.name`                      | -                          | Name of the wallet.                                                                                                  |
|                                 | `--wallet.hotkey`                    | -                  | Hotkey path for the wallet.                                                                                          |
| **PM2 process name**    | `--pm2_name`                        | 'SN16Miner'                          | Name for the pm2 process for Auto Update. |

### License
Refer to the main README for the MIT License details.

