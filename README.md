# Audio Generation Subnetwork

This subnetwork is a decentralized system designed for text-to-audio applications within the Bittensor network. It consists of a Validator and a Miner working collaboratively to generate high-quality audio from provided prompts. 
In the first phase, we will start with text-to-speech (TTS), working in parallel to add music generation in the upcoming updates. 

## Validators

The Validators are responsible for initiating the generation process by providing prompts to the Miners on the network. These prompts serve as the input text for the subsequent TTS model. The Validators evaluates the quality of the generated audio produced by the Miners and rewards them based on the perceived quality.

## Miners

Miners in the Audio Subnetwork are tasked with generating audio from the text prompts received from the Validators. Starting with the leverage of advanced text-to-speech models, miners aim to produce high-fidelity and natural-sounding voice recordings. The quality of the generated audio is crucial, as it directly influences the miner's rewards.

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
cd AudioSubtensor
pip install -r requirements.txt
python -m pip install -e . 
```

**Evaluation Mechanism:**
The evaluation mechanism involves the validator querying miners on the network with random prompts, receiving text-to-speech responses, scoring them based on correctness, and updating weights on the Bittensor network. The scoring is done using a reward function from the `template` module.

**Miner/Validator Hardware Specs:**
The hardware requirements for miners and validators depend on the complexity and resource demands of the chosen text-to-speech models. Typically, a machine with a decent CPU, GPU, sufficient VRAM, and RAM is required. Disk space requirements would depend on the size of the models and any additional data.

**How to Run a Validator:**
To run a validator, execute the `validator.py` script with the necessary command-line arguments. The script sets up Bittensor objects, connects to the network, queries miners, scores responses, and updates weights.

**How to Run a Miner:**
To run a miner, execute the `miner.py` script with the required configuration. The miner initializes Bittensor objects, connects to the network, and processes incoming text-to-speech requests.

**Text-to-Speech Models Supported:**
The code references two text-to-speech models: `TextToSpeechModels` and `SunoBark`. The specific requirements for each model in terms of CPU, GPU VRAM, RAM, and disk space are not explicitly provided in the shared code. To determine these requirements, you may need to refer to the documentation or implementation details of these models.

In general, text-to-speech models can vary in their resource demands. Larger models may require more powerful GPUs and additional system resources. It's recommended to check the documentation or model repository for specific model requirements. If GPU acceleration is utilized, a compatible GPU with sufficient VRAM is often beneficial for faster processing.

Certainly! Below are instructions for using the arguments in `miner.py` and `validator.py`:

### Instructions for `miner.py`:

## Mining 
```bash
python3.9 neurons/miner.py --netuid <your_subnet_uid> --wallet.name <your_wallet_name> --wallet.hotkey default --logging.debug --model <your_model_name>
```
Certainly! Here's the information formatted as a table for README.md:

### Bittensor Miner Script Arguments:

| **Category**                   | **Argument**                         | **Default Value**          | **Description**                                                                                                       |
|---------------------------------|--------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Configuration Arguments**     | `--custom`                           | "my_custom_value"          | Adds a custom value to the parser.                                                                                    |
|                                 | `--model`                            | 'microsoft/speecht5_tts'   | The model to use for text-to-speech.                                                                                 |
|                                 | `--netuid`                           | 1                          | The chain subnet UID.                                                                                                 |
| **Bittensor Subtensor Arguments** | `--subtensor.chain_endpoint`        | -                          | Endpoint for Bittensor chain connection.                                                                              |
|                                 | `--subtensor.network`                | -                          | Bittensor network endpoint.                                                                                          |
| **Bittensor Logging Arguments** | `--logging.debug`                    | -                          | Enable debugging logs.                                                                                               |
|                                 | `--logging.trace`                    | -                          | Enable trace logs.                                                                                                   |
|                                 | `--logging.logging_dir`              | -                          | Directory for logging.                                                                                               |
| **Bittensor Wallet Arguments**  | `--wallet.name`                      | -                          | Name of the wallet.                                                                                                  |
|                                 | `--wallet.hotkey`                    | "default"                  | Hotkey path for the wallet.                                                                                          |
|                                 | `--wallet.path`                      | -                          | Path to the wallet.                                                                                                  |
| **Bittensor Axon Arguments**    | `--axon.port`                        | -                          | Port number for the axon server.                                                                                    |

### Main Function Flow:

1. Set up logging.
2. Check the supplied model and log appropriate information.
3. Initialize Bittensor miner objects (wallet, subtensor, metagraph).
4. Set up miner functionalities (blacklist, priority, main processing function).
5. Build and link miner functions to the axon.
6. Serve the axon on the network with specified netuid.
7. Start the axon server.
8. Keep the miner alive in a loop, periodically updating network knowledge.
9. Handle interruptions and errors gracefully.


## Validating  
```bash
python3.9 neurons/validator.py --netuid <your_subnet_uid> --wallet.name <your_validator_wallet_name> --wallet.hotkey default --logging.debug --hub_key <huggingface_access_key>
```
Certainly! Here's the information formatted as a table for README.md:

### Bittensor Validator Script Arguments:

| **Category**                   | **Argument**                         | **Default Value**          | **Description**                                                                                                       |
|---------------------------------|--------------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Configuration Arguments**     | `--alpha`                            | 0.9                        | The weight moving average scoring.                                                                                    |
|                                 | `--custom`                           | "my_custom_value"          | Adds a custom value to the parser.                                                                                    |
|                                 | `--netuid`                           | 1                          | The chain subnet UID.                                                                                                 |
|                                 | `--hub_key`                          | None                       | Supply the Huggingface Hub API key for the prompt dataset.                                                            |
|                                 | `--threshold`                        | 0.68                       | The threshold for response scoring.                                                                                   |
| **Bittensor Subtensor Arguments** | `--subtensor.chain_endpoint`        | -                          | Endpoint for Bittensor chain connection.                                                                              |
|                                 | `--subtensor.network`                | -                          | Bittensor network endpoint.                                                                                          |
| **Bittensor Logging Arguments** | `--logging.debug`                    | -                          | Enable debugging logs.                                                                                               |
|                                 | `--logging.trace`                    | -                          | Enable trace logs.                                                                                                   |
|                                 | `--logging.logging_dir`              | -                          | Directory for logging.                                                                                               |
| **Bittensor Wallet Arguments**  | `--wallet.name`                      | -                          | Name of the wallet.                                                                                                  |
|                                 | `--wallet.hotkey`                    | "default"                  | Hotkey path for the wallet.                                                                                          |
|                                 | `--wallet.path`                      | -                          | Path to the wallet.                                                                                                  |

### Main Function Flow:

1. Set up logging.
2. Build Bittensor validator objects (wallet, subtensor, dendrite, metagraph).
3. Connect the validator to the network.
4. Set up initial scoring weights for validation.
5. Start the main validation loop.
6. Query miners on the network.
7. Score responses and update weights.
8. Periodically update weights on the Bittensor blockchain.
9. Handle errors and interruptions gracefully.
