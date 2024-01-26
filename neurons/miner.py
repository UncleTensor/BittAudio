# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
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

'''
Disclaimer:
Before using Coqui TTS for commercial purposes, it's essential that you agree to the Coqui license agreement. 
This agreement outlines the terms and conditions for commercial use.
 For more information and to ensure compliance with all legal requirements, 
 please visit their LinkedIn post here. 
 https://www.linkedin.com/posts/coqui-ai_coqui-activity-7095143706399232000--IRi

 '''

# Bittensor Miner lib:

# Step 1: Import necessary libraries and modules
from scipy.io.wavfile import write as write_wav
import bittensor as bt
import numpy as np
import torchaudio
import traceback
import argparse
import typing
import torch
import wave
import time
import sys
import os
import wandb
import platform
import psutil
import GPUtil
import datetime as dt

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

# import this repo
from models.text_to_speech_models import SunoBark, TextToSpeechModels, ElevenLabsTTS, EnglishTextToSpeech
from models.voice_clone import ElevenLabsClone  
from models.bark_voice_clone import BarkVoiceCloning, ModelLoader
import lib.protocol
import lib.utils
import lib


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default= 'microsoft/speecht5_tts' , help="The model to be used for text-to-speech." 
    )
    parser.add_argument(
        "--clone_model", default= 'bark/voiceclone' , help="The model to be used for Voice cloning." 
    )
    parser.add_argument(
        "--eleven_api", default=os.getenv('ELEVEN_API') , help="API key to be used for Eleven Labs." 
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


def main(config):
    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Check the supplied model and log the appropriate information.
    # =========================================== Text To Speech model selection ============================================ 
    try:
        if config.model == "microsoft/speecht5_tts":
            bt.logging.info("Using the TextToSpeechModels with the supplied model: microsoft/speecht5_tts")
            tts_models = TextToSpeechModels()
        elif config.model == "facebook/mms-tts-eng":
            bt.logging.info("Using the English Text-to-Speech with the supplied model: facebook/mms-tts-eng")
            tts_models = EnglishTextToSpeech()
        elif config.model == "suno/bark":
            bt.logging.info("Using the SunoBark with the supplied model: suno/bark")
            tts_models = SunoBark()
        elif config.model == "elevenlabs/eleven" and config.eleven_api is not None:
            bt.logging.info(f"Using the Text-To-Speech with the supplied model: {config.model}")
            tts_models = ElevenLabsTTS(config.eleven_api)
        else:
            bt.logging.error(f"Eleven Labs API key is required for the model: {config.model}")
            exit(1)     
    # =========================================== Text To Speech model selection ============================================
            
    # =========================================== Voice Clone model selection ===============================================    
        if config.clone_model == "bark/voiceclone":
            bt.logging.info("Using the Voice Clone with the supplied model: bark/voiceclone")
            voice_clone_model = ModelLoader()
        elif config.clone_model is not None and config.clone_model == "elevenlabs/eleven" and config.eleven_api is not None:
            bt.logging.info(f"Using the Voice Clone with the supplied model: {config.clone_model}")
            voice_clone_model = ElevenLabsClone(config.eleven_api)
        else:
            bt.logging.error(f"Eleven Labs API key is required for the model: {config.clone_model}")
            exit(1)        
    except Exception as e:
        bt.logging.info(f"An error occurred while model initilization: {e}")
        exit(1)
    # =========================================== Voice Clone model selection ===============================================    

    bt.logging.info("Setting up bittensor objects.")
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour miner: {wallet} is not registered to chain connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()
    
    def get_system_info():
        system_info = {
            "OS -v": platform.platform(),
            "CPU ": os.cpu_count(),
            "RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB", 
        }

        gpus = GPUtil.getGPUs()
        if gpus:
            system_info["GPU"] = gpus[0].name 
        # Convert dictionary to list of strings
        tags = [f"{key}: {value}" for key, value in system_info.items()]
        tags.append(lib.__version__)
        return tags

    use_wandb = True
    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"Miner-{my_subnet_uid}-{run_id}"
    sys_info = get_system_info()

    if use_wandb:
        wandb.init(
            name=name,
            project="AudioSubnet_Miner", 
            entity="subnet16team",
            config={
                "uid": my_subnet_uid,
                "hotkey": wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "miner",
                },
                allow_val_change=True,
                tags=sys_info
            )

############################### Voice Clone ##########################################

    # The blacklist function decides if a request should be ignored.
    def vc_blacklist_fn(synapse: lib.protocol.VoiceClone) -> typing.Tuple[bool, str]:
        #  blacklist( synapse: VoiceClone ) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        elif synapse.dendrite.hotkey in metagraph.hotkeys and metagraph.S[metagraph.hotkeys.index(synapse.dendrite.hotkey)] < lib.MIN_STAKE:
            # Ignore requests from entities with low stake.
            bt.logging.trace(
                f"Blacklisting hotkey {synapse.dendrite.hotkey} with low stake"
            )
            return True, "Low stake"
        elif synapse.dendrite.hotkey in lib.BLACKLISTED_VALIDATORS:
            bt.logging.trace(
                f"Blacklisting Key recognized as blacklisted hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Blacklisted hotkey"
        elif synapse.dendrite.hotkey in lib.WHITELISTED_VALIDATORS:
            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return False, "Hotkey recognized!"
        else:
            bt.logging.trace(
                f"Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Hotkey recognized as Blacklisted!"

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    def vc_priority_fn(synapse: lib.protocol.VoiceClone) -> float:
        caller_uid = metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority
    
    def save_audio(speech):
        '''Save the audio file to disk'''
        try:
            # Check the first few bytes to determine the format
            header = speech[:4]
            if header.startswith(b'RIFF'):
                format = 'WAV'
            elif header.startswith(b'\xFF\xFB') or header.startswith(b'ID3'):
                format = 'MP3'
            else:
                return 'Unknown Format'

            # Generate a file name (You can modify this part as needed)
            new_file_path = 'output_converted.' + format.lower()

            # Save the bytes to a new file
            with open(new_file_path, 'wb') as new_file:
                new_file.write(speech)

            bt.logging.success(f"File has been successfully saved as {new_file_path}")
            return new_file_path
        except Exception as e:
            bt.logging.error(f"Error Occurred while saving the file: {e}")

    
    
    def ElevenlabsClone_call(text, source_file, hf_voice_id):
        '''Call the Eleven Labs API to clone the voice'''
        speech = None
        try:
            speech = voice_clone_model.clone_voice(text, source_file,hf_voice_id)
            elevenlab_file = save_audio(speech)
            return elevenlab_file
        except Exception as e:
            bt.logging.error(f"An error occurred while calling the model: {e}")

    def BarkVoiceClone_call(text, source_file, hf_voice_id):
        '''Call the Bark Voice Clone API to clone the voice'''
        speech = None
        try:
            bvc = BarkVoiceCloning()
            speech = bvc.clone_voice(text, hf_voice_id, source_file, voice_clone_model )
            bark_clone_file_path = "bark_voice_gen.wav"
            write_wav(bark_clone_file_path, rate=24000, data=speech)
            return bark_clone_file_path
        except Exception as e:
            bt.logging.error(f"An error occurred while calling the model: {e}")

    def convert_audio_to_tensor(audio_file):
        '''Convert the audio file to a tensor'''
        try:
            # Get the file extension
            _, file_extension = os.path.splitext(audio_file)

            if file_extension.lower() in ['.wav', '.mp3']:
                # load the audio file
                audio, sample_rate = torchaudio.load(audio_file)
                # convert the audio file to a tensor/list
                audio = audio[0].tolist()
                return audio
            else:
                bt.logging.error(f"Unsupported file format: {file_extension}")
                return None
        except Exception as e:
            bt.logging.error(f"An error occurred while converting the file: {e}")

    def ProcessClone(synapse: lib.protocol.VoiceClone) -> lib.protocol.VoiceClone:
        '''Process the Voice Clone request'''
        bt.logging.debug("The Voice Clone request recieved from validator!")
        speech = None
        try:
            input_text = synapse.text_input
            input_clone = synapse.clone_input
            sample_rate = synapse.sample_rate
            hf_voice_id = synapse.hf_voice_id

            input_tensor = torch.tensor(input_clone, dtype=torch.float32)
            if input_tensor.ndim == 1:
                input_tensor = input_tensor.unsqueeze(0)
            torchaudio.save('input.wav', src=input_tensor, sample_rate=sample_rate)

            # Check if the input text is valid.
            if input_text is None or input_text == "":
                bt.logging.error("No text was supplied. Please supply a valid text.")
                return None
            
            # Check if the input clone is valid.
            if input_clone is None or input_clone == []:
                bt.logging.error("No clone was supplied. Please supply a valid clone.")
                return None
            
        except Exception as e:
            bt.logging.error(f"An error occurred, No input text or input voice recieved: {e}")
            return None

        try:
            if config.clone_model == "elevenlabs/eleven":
                speech_file_path = ElevenlabsClone_call(input_text, 'input.wav',hf_voice_id)
                synapse.model_name = config.clone_model
                speech = convert_audio_to_tensor(speech_file_path)
            elif config.clone_model == "bark/voiceclone":
                speech_file_path = BarkVoiceClone_call(input_text, 'input.wav',hf_voice_id)
                synapse.model_name = config.clone_model
                speech = convert_audio_to_tensor(speech_file_path)
            if speech is not None:
                bt.logging.success(f"Voice Clone has been generated by {config.clone_model}!")
        except Exception as e:
            print(f"An error occurred while clonning the file: {e}")
        
        synapse.clone_output = speech
        return synapse

########################################### Text to Speech ##########################################    


    # The blacklist function decides if a request should be ignored.
    def speech_blacklist_fn(synapse: lib.protocol.TextToSpeech) -> typing.Tuple[bool, str]:
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        elif synapse.dendrite.hotkey in metagraph.hotkeys and metagraph.S[metagraph.hotkeys.index(synapse.dendrite.hotkey)] < lib.MIN_STAKE:
            # Ignore requests from entities with low stake.
            bt.logging.trace(
                f"Blacklisting hotkey {synapse.dendrite.hotkey} with low stake"
            )
            return True, "Low stake"
        elif synapse.dendrite.hotkey in lib.BLACKLISTED_VALIDATORS:
            bt.logging.trace(
                f"Blacklisting Key recognized as blacklisted hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Blacklisted hotkey"
        elif synapse.dendrite.hotkey in lib.WHITELISTED_VALIDATORS:
            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return False, "Hotkey recognized!"
        else:
            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Hotkey recognized as Blacklisted!"

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    def speech_priority_fn(synapse: lib.protocol.TextToSpeech) -> float:
        caller_uid = metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    def ProcessSpeech(synapse: lib.protocol.TextToSpeech) -> lib.protocol.TextToSpeech:
        bt.logging.success("The prompt received from validator!")
        if config.model == "microsoft/speecht5_tts":
            speech = tts_models.generate_speech(synapse.text_input)
        elif config.model == "elevenlabs/eleven":
            speech = tts_models.generate_speech(synapse.text_input)
        elif config.model == "suno/bark":
            speech = tts_models.generate_speech(synapse.text_input)
        elif config.model == "facebook/mms-tts-eng":
            speech = tts_models.generate_speech(synapse.text_input)
            audio_data = speech / torch.max(torch.abs(speech))

            # If the audio is mono, ensure it has a channel dimension
            if audio_data.ndim == 1:
                audio_data = audio_data.unsqueeze(0)

            # convert to 32-bit PCM
            audio_data_int = (audio_data * 2147483647).type(torch.IntTensor)

            # Save the audio data as integers
            torchaudio.save('speech.wav', src=audio_data_int, sample_rate=16000)
            # Open the WAV file and read the frames
            sample_width = None
            try:
                with wave.open('speech.wav', 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    sample_width = wav_file.getsampwidth()
            except Exception as e:
                print(f"An error occurred while reading the audio data: {e}")
            # Initialize dtype to a default value
            dtype = None
            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 1:
                dtype = np.int8
            elif sample_width == 4:
                dtype = np.int32

            # Check if dtype has been assigned a value
            if dtype is None:
                print(f"Unexpected sample width: {sample_width}")
                return

            # Convert the bytes data to a numpy array
            audio_array = np.frombuffer(frames, dtype=dtype)
            # Convert the numpy array to a list
            speech = audio_array.tolist()

        # Check if 'speech' contains valid audio data
        if speech is None:
            bt.logging.error("No speech generated!")
            return None
        else:
            try:
                bt.logging.success(f"Text to Speech has been generated by {config.model}!")
                if config.model == "facebook/mms-tts-eng":
                    # Convert the list to a tensor
                    speech_tensor = torch.Tensor(speech)

                    # Normalize the speech data
                    audio_data = speech_tensor / torch.max(torch.abs(speech_tensor))

                    # Convert to 32-bit PCM
                    audio_data_int = (audio_data * 2147483647).type(torch.IntTensor)

                    # Add an extra dimension to make it a 2D tensor
                    audio_data_int = audio_data_int.unsqueeze(0)

                    # Save the audio data as a .wav file
                    synapse.speech_output = speech  # Convert PyTorch tensor to a list

                elif config.model == "suno/bark":
                    speech = speech.cpu().numpy().squeeze()
                    synapse.model_name = config.model
                    synapse.speech_output = speech.tolist()

                elif config.model == "elevenlabs/eleven":
                    speech_file = save_audio(speech)
                    synapse.model_name = config.model
                    speech = convert_audio_to_tensor(speech_file)
                    synapse.speech_output = speech
                else:
                    
                    synapse.speech_output = speech.tolist()  # Convert PyTorch tensor to a list
                return synapse
            except Exception as e:
                print(f"An error occurred while processing speech output: {e}")

####################################################### Attach Axon  ##############################################################
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon(wallet=wallet, config=config)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn= ProcessClone, 
        blacklist_fn= vc_blacklist_fn, 
        priority_fn= vc_priority_fn).attach(
        forward_fn= ProcessSpeech,
        blacklist_fn= speech_blacklist_fn,
        priority_fn= speech_priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving axon {ProcessSpeech} on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
    )
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 7: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # TODO(developer): Define any additional operations to be performed by the miner.
            # Below: Periodically update our knowledge of the network graph.
            if step % 500 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[my_subnet_uid]:.6f} | "
                    f"Rank:{metagraph.R[my_subnet_uid]:.6f} | "
                    f"Trust:{metagraph.T[my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[my_subnet_uid]:.6f} | "
                    f"Incentive:{metagraph.I[my_subnet_uid]:.6f} | "
                    f"Emission:{metagraph.E[my_subnet_uid]}"
                )
                bt.logging.info(log)
            step += 1
            time.sleep(1)

            if step % 1000 == 0:
                lib.utils.try_update()

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            wandb.finish()
            bt.logging.success("Wandb finished.")
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
# Entry point for the script
if __name__ == "__main__":
    config = get_config()
    main(config)
