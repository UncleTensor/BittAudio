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
        if np.max(np.abs(audio_signal)) < silence_threshold or np.var(audio_signal) < constant_signal_threshold:
            return 0
        signal_power = np.mean(audio_signal**2)
        noise_signal = librosa.effects.preemphasis(audio_signal)
        noise_power = np.mean(noise_signal**2)
        if noise_power < 1e-10:
            return 0
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def calculate_hnr(file_path):
        """
        Harmonic to noise ratio is a measure of the relations between tone and noise.
        A high value means less noise, a low value means more noise.
        """
        y, _ = librosa.load(file_path, sr=None)
        if np.max(np.abs(y)) < 1e-4 or np.var(y) < 1e-2:
            return 0
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_power = np.mean(harmonic**2)
        noise_power = np.mean(percussive**2)
        hnr = 10 * np.log10(harmonic_power / max(noise_power, 1e-10))
        return hnr

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

class Normalizer:
    @staticmethod
    def normalize_quality(quality_metric):
        # Normalize quality to be within 0 to 1, with good values above 20 dB considered as high quality
        return 1 / (1 + np.exp(-((quality_metric - 20) / 10)))

    @staticmethod
    def normalize_consistency(score):
        if score is not None:
            if score > 0:
                normalized_consistency = (score + 1) / 2
            else:
                normalized_consistency = 0
        else:
            normalized_consistency = 0
        return normalized_consistency

class Aggregator:
    @staticmethod
    def geometric_mean(scores):
        """Calculate the geometric mean of the scores, avoiding any non-positive values."""
        scores = [max(score, 0.0001) for score in scores.values()]  # Replace non-positive values to avoid math errors
        product = np.prod(scores)
        return product ** (1.0 / len(scores))

class MusicQualityEvaluator:
    def __init__(self):
        self.metric_evaluator = MetricEvaluator()
        self.normalizer = Normalizer()
        self.aggregator = Aggregator()

    def evaluate_music_quality(self, file_path, text=None):
        try:
            snr_score = self.metric_evaluator.calculate_snr(file_path)
            bt.logging.info(f'.......SNR......: {snr_score} dB')
        except:
            pass
            bt.logging.error(f"Failed to calculate SNR")

        try:
            hnr_score = self.metric_evaluator.calculate_hnr(file_path)
            bt.logging.info(f'.......HNR......: {hnr_score} dB')
        except:
            pass
            bt.logging.error(f"Failed to calculate SNR")

        try:
            consistency_score = self.metric_evaluator.calculate_consistency(file_path, text)
            bt.logging.info(f'....... Consistency Score ......: {consistency_score}')
        except:
            pass
            bt.logging.error(f"Failed to calculate Consistency score")

        # Normalize scores and calculate aggregate score
        normalized_snr = self.normalizer.normalize_quality(snr_score)
        normalized_hnr = self.normalizer.normalize_quality(hnr_score)
        normalized_consistency = self.normalizer.normalize_consistency(consistency_score)

        bt.logging.info(f'Normalized Metrics: SNR = {normalized_snr}dB, Normalized Metrics: HNR = {normalized_hnr}dB, Consistency = {normalized_consistency}')
        aggregate_quality = self.aggregator.geometric_mean({'snr': normalized_snr, 'hnr': normalized_hnr})
        aggregate_score = self.aggregator.geometric_mean({'quality': aggregate_quality, 'normalized_consistency': normalized_consistency}) if consistency_score > 0.2 else 0
        bt.logging.info(f'....... Aggregate Score ......: {aggregate_score}')
        return aggregate_score