# BittAudio (SN 50) | Audio Generation Subnet on Bittensor
![bitaudio](docs/bittaudio.jpg)
The main goal of the BittAudio is to establish a decentralized platform that incentivizes the creation, distribution and also monetization of AI audio content, such as:
- Text-to-Music (TTM) <br>

Validators and miners work together to ensure high-quality outputs, fostering innovation and rewarding contributions in the audio domain.<br>
By introducing audio generation service such as Text-to-Music, this subnetwork expands the range of available service within the Bittensor ecosystem. This diversification enhances the utility and appeal of the Bittensor platform to a broader audience, including creators, influencers, developers, and end-users interested in audio content.<br><br>

## Validators & Miners Interaction
- Validators initiate requests filled with the required data and encrypt them with a symmetric key. 
- Requests are signed with the validator’s private key to certify authenticity. 
- Miners decrypt the requests, verify the signatures to ensure authenticity, process the requests, and then send back the results, encrypted and signed for security.

**Validators** are responsible for initiating the generation process by providing prompts to the Miners on the network. These prompts serve as the input for TTM service. The Validators then evaluate the quality of the generated audio and reward the Miners based on the output quality.<br>
Please refer to the [Validator Documentation](docs/validator.md)

**Miners** in the Audio Subnetwork are tasked with generating audio from the text prompts received from the Validators. Leveraging advanced TTM models, miners aim to produce high-fidelity music melodies. The quality of the generated audio is crucial, as it directly influences the miners' rewards.<br>
Please refer to the [Miner Documentation](docs/miner.md)

## Workflow

1. **Prompt Generation:** The Validators generates TTM prompts and distributes them to the Miners on the network.

2. **Audio Processing:** Miners receive the prompts and utilize TTM models to convert the text into audio (music).

3. **Quality Evaluation:** The Validator assesses the quality of the generated audio, considering factors such as: clarity, naturalness, and adherence to the prompt.

4. **Reward Distribution:** Based on the quality assessment, the Validator rewards Miners accordingly. Miners with consistently higher-quality outputs receive a larger share of rewards.

## Data Sources
To combat potential data exhaustion and ensure uniqueness, our subnet has integrated with the Corcel API, part of Bittensor Subnetwork 18. This integration allows Validators to generate synthetic, unique prompts, significantly reducing redundancy risks and improving network security, thus our subnet has taken a step towards strengthening the Bittensor ecosystem.<br><br>
**SN 18 - Cortex.t (Corcel API):**<br>
https://docs.corcel.io/reference/cortext-text

To ensure there is no downtime in Corcel API, Validators utilize a diverse dataset, selecting prompts randomly from a pool of 500K prompts of TTM service.

**Data used from HuggingFace for TTM:**<br>
•	TTM music generation prompts (currently 500K) <br>
https://huggingface.co/datasets/etechgrid/prompts_for_TTM <br>

## Collaborations
**BittAudio currently collaborate with one Bittensor subnetworks.** <br><br>
**SN 18 - Cortex.t (using Corcel.io API)** <br>
We have integrated with SN 18 through Corcel API to utilize synthetic, human-like data for service: TTM. <br>
https://docs.corcel.io/reference/cortext-text
## Applications

## Benefits

- **Decentralized Text-to-Audio:** The subnetwork decentralizes the Text-to-Music process, distributing the workload among participating Miners.
  
- **Quality Incentives:** The incentive mechanism encourages Miners to continually improve the quality of their generated audio.

- **Bittensor Network Integration:** Leveraging the Bittensor network ensures secure and transparent interactions between Validators and Miners.

Join BittAudio and contribute to the advancement of decentralized Text-to-Music technology within the Bittensor ecosystem.


## Installation
```bash 
git clone https://github.com/UncleTensor/BittAudio.git
cd BittAudio
pip install -e .
pip install -r requirements.txt
wandb login
```

## Recommended GPU Configuration

It is recommended to use NVIDIA GeForce RTX A6000 GPUs at minimum for both Validators and Miners.


**Evaluation Mechanism:** <br>
The evaluation mechanism involves the Validators querying miners on the network with random prompts and receiving TTM responses. These responses are scored based on correctness, and the weights on the Bittensor network are updated accordingly. The scoring is conducted using a reward function from the lib module.

**Miner/Validator Hardware Specs:**<br>
The hardware requirements for miners and validators vary depending on the complexity and resource demands of the selected TTM models. Typically, a machine equipped with a capable CPU and GPU, along with sufficient VRAM and RAM, is necessary. The amount of disk space required will depend on the size of the models and any additional data.

**How to Run a Validator:**<br>
To operate a validator, you need to run the validator.py script with the required command-line arguments. This script initiates the setup of Bittensor objects, establishes a connection to the network, queries miners, scores their responses, and updates weights accordingly.

**How to Run a Miner:**<br>
To operate a miner, run the miner.py script with the necessary configuration. This process involves initializing Bittensor objects, establishing a connection to the network, and processing incoming TTM requests.

**TTM Models Supported:**<br>
The code incorporates various Text-to-Music models. The specific requirements for each model, including CPU, GPU VRAM, RAM, and disk space, are not explicitly stated in the provided code. For these type of requirements, it may be necessary to consult the documentation or delve into the implementation details of these models.

In general, the resource demands of TTM models can vary significantly. Larger models often necessitate more powerful GPUs and additional system resources. It is advisable to consult the documentation or model repository for the specific requirements of each model. Additionally, if GPU acceleration is employed, having a compatible GPU with enough VRAM is typically advantageous for faster processing.

## License
This repository is licensed under the MIT License.

```text
MIT License

Copyright (c) 2024 Opentensor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
