# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
# Copyright © 2023 ETG

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
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import AutoProcessor, AutoModelForTextToWaveform
from speechbrain.pretrained import EncoderClassifier
from transformers import VitsModel, AutoTokenizer
from torchaudio.transforms import Resample
from datasets import load_dataset
import torchaudio
import torch
import os


# Speaker Embedding Generator using SpeechBrain's Speaker Recognition Model
class SpeakerRecognizer:
    def __init__(self):
        self.spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker_model = EncoderClassifier.from_hparams(
            source=self.spk_model_name,
            run_opts={"device": self.device},
            savedir=os.path.join("/tmp", self.spk_model_name),
        )

    def create_speaker_embedding(self, audio_file_path):
        waveform, sample_rate = torchaudio.load(audio_file_path)

        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(waveform)
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()

        return speaker_embeddings
    
# Text-to-Speech Synthesis with Customizable Speaker Embeddings
class TextToSpeechModels:
    def __init__(self):
        # Load the models and processor
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Initialize SpeakerRecognizer
        self.speaker_recognizer = SpeakerRecognizer()

    def generate_speech(self, text_input, audio_file_path=None):
        # Process the text input
        inputs = self.processor(text=text_input, return_tensors="pt")

        # Generate speaker embeddings from the provided audio file
        if audio_file_path:
            speaker_embeddings = self.speaker_recognizer.create_speaker_embedding(audio_file_path = 'sample3.wav')
        else:
            # Use default embeddings if no audio file is provided
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        # Generate speech
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
        return speech


# Text-to-Speech Generation Using Suno Bark's Pretrained Model
class SunoBark:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = AutoModelForTextToWaveform.from_pretrained("suno/bark-small")

    def generate_speech(self, text_input,):
        inputs = self.processor(
            text=text_input,
            return_tensors="pt",
            padding="longest",  # Pad to the longest sequence in the batch (or a specified length)
            add_special_tokens=True  # Add special tokens (e.g., [CLS], [SEP])
        )
        # Ensure 'attention_mask' is included in the inputs
        speech = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            do_sample=True

        )
        return speech
    

# from scipy.io.wavfile import write as write_wav

# class EnglishTextToSpeech:
#     def __init__(self, output_file="output.wav"):
#         self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
#         self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
#         self.output_file = output_file

#     def generate_speech(self, text_input):
#         inputs = self.tokenizer(text_input, return_tensors="pt")
#         with torch.no_grad():
#             output = self.model(**inputs).waveform

#         # Save the waveform as a WAV file
#         write_wav(self.output_file, rate=self.model.config.sampling_rate, data=output.float().numpy())
        
#         return output