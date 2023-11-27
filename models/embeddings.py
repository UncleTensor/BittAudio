import os
import torch
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.pretrained import EncoderClassifier

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
