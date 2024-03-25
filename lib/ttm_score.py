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
    def calculate_consistency(file_path, text):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pt_file = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt")
            clap_metric = CLAPTextConsistencyMetric(pt_file, model_arch='HTSAT-base').to(device)
            def convert_audio(audio, from_rate, to_rate, to_channels):
                resampler = torchaudio.transforms.Resample(orig_freq=from_rate, new_freq=to_rate)
                audio = resampler(audio)
                if to_channels == 1:
                    audio = audio.mean(dim=0, keepdim=True)
                return audio

            audio, sr = torchaudio.load(file_path)
            audio = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=1)

            clap_metric.update(audio.unsqueeze(0), [text], torch.tensor([audio.shape[1]]), torch.tensor([sr]))
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
            consistency_score = MetricEvaluator.calculate_consistency(file_path, text)
            bt.logging.info(f'....... Consistency Score ......: {consistency_score}')
        except:
            bt.logging.error(f"Failed to calculate Consistency score")

        # Normalize scores and calculate aggregate score
        normalized_snr = 1 / (1 + np.exp(-snr_score / 20))
        normalized_consistency = (consistency_score + 1) / 2 if consistency_score is not None and consistency_score >= 0 else 0

        if consistency_score is not None:
            if consistency_score > 0:
                normalized_consistency = (consistency_score + 1) / 2  # Normalizes from [0, 1] to [0.5, 1]
            else:
                normalized_consistency = 0  # Ensures that a consistency_score of 0 or any negative value yields a normalized score of 0
        else:
            normalized_consistency = 0  # Handles cases where consistency_score is None

        bt.logging.info(f'....... Normalized SNR {normalized_snr}DB - Normalized Consistency {normalized_consistency} ......')
        bt.logging.info(f'....... SNR {snr_score}DB - Consistency {consistency_score} ......')
        aggregate_score = 0.6 * normalized_snr + 0.4 * normalized_consistency 
        aggregate_score = aggregate_score if consistency_score >= 0.2 else 0
        bt.logging.info(f'....... Aggregate Score ......: {aggregate_score}')
        return aggregate_score