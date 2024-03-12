from huggingface_hub import hf_hub_download
import numpy as np
import librosa
import torch
import torchaudio
from scipy.signal import hilbert
from audiocraft.metrics import CLAPTextConsistencyMetric
import bittensor as bt


class MetricEvaluator:
    @staticmethod
    def calculate_snr(file_path, silence_threshold=1e-4, constant_signal_threshold=1e-2):
        audio_signal, _ = librosa.load(file_path, sr=None)
        if np.max(np.abs(audio_signal)) < silence_threshold:
            return -np.inf
        elif np.var(audio_signal) < constant_signal_threshold:
            return -np.inf
        signal_power = np.mean(audio_signal**2)
        noise_signal = librosa.effects.preemphasis(audio_signal)
        noise_power = np.mean(noise_signal**2)
        if noise_power < 1e-10:
            return np.inf
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def calculate_smoothness(file_path, silence_threshold=1e-10):
        audio_signal, _ = torchaudio.load(file_path)
        if torch.max(audio_signal) < silence_threshold:
            return 0
        amplitude_envelope = torch.abs(torch.from_numpy(np.abs(hilbert(audio_signal[0].numpy()))))
        amplitude_differences = torch.abs(amplitude_envelope[1:] - amplitude_envelope[:-1])
        smoothness = torch.mean(amplitude_differences)
        return smoothness.item()

    @staticmethod
    def calculate_consistency(file_path, text):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pt_file = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-best.pt")
            clap_metric = CLAPTextConsistencyMetric(pt_file).to(device)
            def convert_audio(audio, from_rate, to_rate, to_channels):
                resampler = torchaudio.transforms.Resample(orig_freq=from_rate, new_freq=to_rate)
                audio = resampler(audio)
                if to_channels == 1:
                    audio = audio.mean(dim=0, keepdim=True)
                return audio

            audio, sr = torchaudio.load(file_path)
            audio = convert_audio(audio, from_rate=sr, to_rate=48000, to_channels=1)

            clap_metric.update(audio.unsqueeze(0), [text], torch.tensor([audio.shape[1]]), torch.tensor([48000]))
            consistency_score = clap_metric.compute()
            return consistency_score
        except Exception as e:
            print(f"An error occurred while calculating music consistency score: {e}")
            return None

class MusicQualityEvaluator:
    def __init__(self):
        pass

    def evaluate_music_quality(self, file_path, text=None):
        try:
            snr_score = MetricEvaluator.calculate_snr(file_path)
            bt.logging.info(f'.......SNR......: {snr_score} dB')
        except:
            bt.logging.error(f"Failed to calculate SNR")

        try:
            smoothness_score = MetricEvaluator.calculate_smoothness(file_path)
            bt.logging.info(f'.......Smoothness Score......: {smoothness_score}')
        except:
            bt.logging.error(f"Failed to calculate Smoothness score")

        try:
            consistency_score = MetricEvaluator.calculate_consistency(file_path, text)
            bt.logging.info(f'.......Consistency Score......: {consistency_score}')
        except:
            bt.logging.error(f"Failed to calculate Consistency score")

        # Normalize scores and calculate aggregate score
        normalized_snr = 1 / (1 + np.exp(-snr_score / 20))
        normalized_smoothness = 1 - smoothness_score if smoothness_score is not None else 1
        normalized_consistency = (consistency_score + 1) / 2 if consistency_score is not None and consistency_score >= 0 else 0

        aggregate_score = (normalized_snr + normalized_smoothness + normalized_consistency) / 3.0 if consistency_score >=0 else 0
        bt.logging.info(f'.......Aggregate Score......: {aggregate_score}')
        return aggregate_score