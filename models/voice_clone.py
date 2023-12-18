from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
from torchaudio.transforms import Resample
from datasets import load_dataset
import soundfile as sf
import torchaudio
import torch
import os


class MicrosoftVoiceCloner:
    def __init__(self):
        self.processor = None
        self.model = None
        self.vocoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker_model = None

    def load_models(self, tts_model_name="microsoft/speecht5_tts", vocoder_model_name="microsoft/speecht5_hifigan", spk_model_name="speechbrain/spkrec-xvect-voxceleb"):
        self.processor = SpeechT5Processor.from_pretrained(tts_model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_name)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_name)
        self.speaker_model = EncoderClassifier.from_hparams(
            source=spk_model_name,
            run_opts={"device": self.device},
            savedir=os.path.join("/tmp", spk_model_name),
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

            if len(speaker_embeddings.shape) == 2:
                speaker_embeddings = speaker_embeddings.mean(axis=0)

        return speaker_embeddings

    def generate_speech(self, text, speaker_embeddings, output_path):
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
        return sf.write(output_path, speech.numpy(), samplerate=16000)