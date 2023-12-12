# Audio Generation Subnetwork

This subnetwork is a decentralized system designed for text-to-audio applications within the Bittensor network. It consists of a Validator and a Miner working collaboratively to generate high-quality audio from provided prompts. 
In the first phase, we will start with text-to-speech (TTS), working in parallel to add music generation in the upcoming updates. 

## Validators

The Validators are responsible for initiating the generation process by providing prompts to the Miners on the network. These prompts serve as the input text for the subsequent TTS model. The Validators evaluate the quality of the generated audio produced by the Miners and reward them based on the perceived quality.

## Miners

Miners in the Audio Subnetwork are tasked with generating audio from the text prompts received from the Validators. Leveraging advanced text-to-speech models, miners aim to produce high-fidelity, natural-sounding voice recordings. The quality of the generated audio is crucial, as it directly influences the miners' rewards.

## Workflow

1. **Prompt Generation:** The Validators generates prompts and distributes them to the Miners on the network.

2. **Text-to-Speech Processing:** Miners receive the prompts and utilize text-to-speech models to convert the text into voice audio.

3. **Quality Evaluation:** The Validator assesses the quality of the generated audio, considering factors such as: clarity, naturalness, and adherence to the prompt.

4. **Reward Distribution:** Based on the quality assessment, the Validator rewards Miners accordingly. Miners with consistently higher-quality outputs receive a larger share of rewards.

## Benefits

- **Decentralized Text-to-Speech:** The subnetwork decentralizes the TTS process, distributing the workload among participating Miners.
  
- **Quality Incentives:** The incentive mechanism encourages Miners to continually improve the quality of their generated voice audio.

- **Bittensor Network Integration:** Leveraging the Bittensor network ensures secure and transparent interactions between Validators and Miners.

Join the Audio Subnetwork and contribute to the advancement of decentralized text-to-speech / text-to-music technologies within the Bittensor ecosystem.


## Installation
```bash 
git clone https://github.com/UncleTensor/AudioSubnet.git
cd AudioSubnet
pip install -r requirements.txt
python -m pip install -e . 
```

## Recommended GPU Configuration

For optimal performance, it is recommended to use NVIDIA GeForce RTX 3090 GPUs for both Validators and Miners.


**Evaluation Mechanism:**
The evaluation mechanism involves the validator querying miners on the network with random prompts and receiving text-to-speech responses. These responses are scored based on correctness, and the weights on the Bittensor network are updated accordingly. The scoring is conducted using a reward function from the lib module.

**Miner/Validator Hardware Specs:**
The hardware requirements for miners and validators vary depending on the complexity and resource demands of the selected text-to-speech models. Typically, a machine equipped with a capable CPU and GPU, along with sufficient VRAM and RAM, is necessary. The amount of disk space required will depend on the size of the models and any additional data.

**How to Run a Validator:**
To operate a validator, you need to run the validator.py script with the required command-line arguments. This script initiates the setup of Bittensor objects, establishes a connection to the network, queries miners, scores their responses, and updates weights accordingly.

**How to Run a Miner:**
To operate a miner, run the miner.py script with the necessary configuration. This process involves initializing Bittensor objects, establishing a connection to the network, and processing incoming text-to-speech requests.

**Text-to-Speech Models Supported:**
The code incorporates three text-to-speech models: Microsoft/speecht5_tts, Facebook/mms-tts-eng and SunoBark. However, the specific requirements for each model, including CPU, GPU VRAM, RAM, and disk space, are not explicitly stated in the provided code. To ascertain these requirements, it may be necessary to consult the documentation or delve into the implementation details of these models.

In general, the resource demands of text-to-speech models can vary significantly. Larger models often necessitate more powerful GPUs and additional system resources. It is advisable to consult the documentation or model repository for the specific requirements of each model. Additionally, if GPU acceleration is employed, having a compatible GPU with enough VRAM is typically advantageous for faster processing.

## Auto-Update Feature

The Auto-Update feature in the Audio Subnetwork ensures that the software components, including text-to-speech models and other dependencies, remain up-to-date with the latest improvements and bug fixes. This feature is designed to:

1. **Automatically Check for Updates:** Periodically scans for new updates or patches available in the repository.
2. **Seamless Integration:** Integrates updates without disrupting ongoing processes or requiring manual intervention.
3. **Enhanced Security and Performance:** By keeping the software up-to-date, it ensures enhanced security and optimal performance.
4. **User Notification:** Notifies users about the updates being installed, allowing them to be aware of the changes and improvements.

To enable this feature, ensure you `--auto_update` argument with the value `yes`

Below are instructions for using the arguments in `miner.py` and `validator.py`:

### Instructions for `miner.py`:

## Mining 
```bash
python3 neurons/miner.py --netuid <subnet_uid> --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --logging.debug --model <model_name>
```

### Bittensor Miner Script Arguments:

| **Category**                   | **Argument**                         | **Default Value**          | **Description**                                                                                                       |
|---------------------------------|--------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
|                                 | `--model`                            | 'microsoft/speecht5_tts' ; 'facebook/mms-tts-eng' ; 'suno/bark'   | The model to use for text-to-speech.                                                                                 |
|                                 | `--netuid`                           | Testnet: 31 ; Mainnet: 16                          | The chain subnet UID.                                                                                                 |
| **Bittensor Subtensor Arguments** | `--subtensor.chain_endpoint`        | -                          | Endpoint for Bittensor chain connection.                                                                              |
|                                 | `--subtensor.network`                | -                          | Bittensor network endpoint.                                                                                          |
| **Bittensor Logging Arguments** | `--logging.debug`                    | -                          | Enable debugging logs.                                                                                               |
|                                 | `--logging.trace`                    | -                          | Enable trace logs.                                                                                                   |
|                                 | `--logging.logging_dir`              | -                          | Directory for logging.                                                                                               |
| **Bittensor Wallet Arguments**  | `--wallet.name`                      | -                          | Name of the wallet.                                                                                                  |
|                                 | `--wallet.hotkey`                    | -                  | Hotkey path for the wallet.                                                                                          |
|                                 | `--wallet.path`                      | -                          | Path to the wallet.                                                                                                  |
| **Bittensor Axon Arguments**    | `--axon.port`                        | -                          | Port number for the axon server.                                                                                    |
| **Auto Update Argument**       | `--auto_update`                        | yes                          | Updates the repository                                                                                     |


### Instructions for `validator.py`:

## Validating  
```bash
python3 neurons/validator.py --netuid <subnet_uid> --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --logging.debug --hub_key <huggingface_access_key>
```

### Bittensor Validator Script Arguments:

| **Category**                   | **Argument**                         | **Default Value**          | **Description**                                                                                                       |
|---------------------------------|--------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Configuration Arguments**     | `--alpha`                            | 0.9                        | The weight moving average scoring.                                                                                    |
|                                 | `--custom`                           | "my_custom_value"          | Adds a custom value to the parser.                                                                                    |
|                                 | `--netuid`                           | Testnet: 31 ; Mainnet: 16                          | The chain subnet UID.                                                                                                 |
|                                 | `--hub_key`                          | None                       | Supply the Huggingface Hub API key for the prompt dataset.                                                            |
| **Bittensor Subtensor Arguments** | `--subtensor.chain_endpoint`        | -                          | Endpoint for Bittensor chain connection.                                                                              |
|                                 | `--subtensor.network`                | -                          | Bittensor network endpoint.                                                                                          |
| **Bittensor Logging Arguments** | `--logging.debug`                    | -                          | Enable debugging logs.                                                                                               |
|                                 | `--logging.trace`                    | -                          | Enable trace logs.                                                                                                   |
|                                 | `--logging.logging_dir`              | -                          | Directory for logging.                                                                                               |
| **Bittensor Wallet Arguments**  | `--wallet.name`                      | -                          | Name of the wallet.                                                                                                  |
|                                 | `--wallet.hotkey`                    | -                  | Hotkey path for the wallet.                                                                                          |
|                                 | `--wallet.path`                      | -                          | Path to the wallet.                                                                                                  |
| **Auto Update Argument**       | `--auto_update`                        | yes                          | Updates the repository                                                                                     |


