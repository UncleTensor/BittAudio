from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
import librosa
import torch
import torchaudio
from scipy.signal import hilbert
from pathlib import Path
from audiocraft.metrics import CLAPTextConsistencyMetric
import subprocess
import os

class MetricEvaluator:
    @staticmethod
    def calculate_snr(file_path):
        audio_signal, _ = librosa.load(file_path, sr=None)
        signal_power = np.sum(audio_signal**2)
        noise_power = np.sum(librosa.effects.preemphasis(audio_signal)**2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def calculate_smoothness(file_path):
        audio_signal, _ = torchaudio.load(file_path)
        amplitude_envelope = torch.from_numpy(np.abs(hilbert(audio_signal[0].numpy())))
        smoothness = 0.0
        smoothness = torch.sum(torch.abs(amplitude_envelope[1:] - amplitude_envelope[:-1]))
        smoothness /= len(amplitude_envelope) - 1
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
            print(f"An error occurred -------------------: {e}")
            return None
    



class MusicQualityEvaluator:
    def __init__(self):
        pass

    def evaluate_music_quality(self, file_path, text=None):
        try:
            snr_value = MetricEvaluator.calculate_snr(file_path)
            print(f'SNR: {snr_value} dB')
        except:
            print("SNR could not be calculated")

        try:
            smoothness_score = MetricEvaluator.calculate_smoothness(file_path)
            print(f'Smoothness Score: {smoothness_score}')
        except:
            print("Smoothness could not be calculated")

        try:
            consistency_score = MetricEvaluator.calculate_consistency(file_path, text)
            print(f"Consistency Score: {consistency_score}")
        except:
            print("Consistency could not be calculated")

        # Normalize scores and calculate aggregate score
        normalized_snr = snr_value / 20.0
        normalized_smoothness = 1 - smoothness_score if smoothness_score is not None else 0
        normalized_consistency = (consistency_score + 1) / 2 if consistency_score is not None else 0

        aggregate_score = (normalized_snr + normalized_smoothness + normalized_consistency) / 3.0
        print(f"Aggregate Score: {aggregate_score}")
        return aggregate_score
