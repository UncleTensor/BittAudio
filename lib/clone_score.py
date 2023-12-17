import torch
import torchaudio
import torchaudio.transforms as T
from reward import score

class AudioProcessor:
    def __init__(self, n_mels=128):
        self.n_mels = n_mels

    def extract_mel_spectrogram(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=self.n_mels)
        mel_spectrogram = mel_spectrogram_transform(waveform)
        return mel_spectrogram

    def pad_or_trim_to_same_length(self, spec1, spec2):
        if spec1.size(2) > spec2.size(2):
            padding_size = spec1.size(2) - spec2.size(2)
            spec2 = torch.nn.functional.pad(spec2, (0, padding_size))
        elif spec2.size(2) > spec1.size(2):
            padding_size = spec2.size(2) - spec1.size(2)
            spec1 = torch.nn.functional.pad(spec1, (0, padding_size))
        return spec1, spec2

    def calculate_mse(self, spec1, spec2):
        return torch.mean((spec1 - spec2) ** 2)

    def compare_audio(self, file_path1, file_path2, input_text=None):
        # Extract Mel Spectrograms
        spec1 = self.extract_mel_spectrogram(file_path1)
        spec2 = self.extract_mel_spectrogram(file_path2)

        # Pad or Trim
        spec1, spec2 = self.pad_or_trim_to_same_length(spec1, spec2)

        # Calculate MSE
        mse_score = self.calculate_mse(spec1, spec2).item
        _score = score(file_path2, input_text)
        # Adjust MSE Score
        if mse_score < 1:
            adjusted_mse = 1
        elif mse_score < 5:
            adjusted_mse = 0.9
        elif mse_score < 10:
            adjusted_mse = 0.8
        elif mse_score < 15:
            adjusted_mse = 0.7
        elif mse_score < 20:
            adjusted_mse = 0.6
        else:
            adjusted_mse = 0.5

        return (adjusted_mse + _score)/2
# Usage
audio_processor = AudioProcessor()
mse_score = audio_processor.compare_audio('path_to_audio1.wav', 'path_to_audio2.wav')
print("MSE Score:", mse_score)
