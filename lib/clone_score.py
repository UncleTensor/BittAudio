import torch
import torchaudio
import torchaudio.transforms as T
import bittensor as bt
from lib.reward import score
import math
import numpy as np
from scipy.spatial.distance import cosine
from torchaudio.transforms import Vad

class CloneScore:
    def __init__(self, n_mels=128):
        self.n_mels = n_mels
        self.vad = Vad(sample_rate=16000)  # Voice Activity Detection for trimming silence

    def trim_silence(self, waveform,):
        # Assuming the audio is mono for simplicity; adjust or expand as needed for your use case
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        trimmed_waveform = self.vad(waveform)
        return trimmed_waveform

    def extract_mel_spectrogram(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        # Trim silence from the waveform
        waveform = self.trim_silence(waveform)
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=self.n_mels)
        mel_spectrogram = mel_spectrogram_transform(waveform)
        # Convert power spectrogram to dB units and normalize
        db_transform = T.AmplitudeToDB()
        mel_spectrogram_db = db_transform(mel_spectrogram)
        norm_spectrogram = (mel_spectrogram_db - mel_spectrogram_db.mean()) / mel_spectrogram_db.std()
        return norm_spectrogram

    def pad_or_trim_to_same_length(self, spec1, spec2):
        if spec1.size(2) > spec2.size(2):
            padding_size = spec1.size(2) - spec2.size(2)
            spec2 = torch.nn.functional.pad(spec2, (0, padding_size))
        elif spec2.size(2) > spec1.size(2):
            padding_size = spec2.size(2) - spec1.size(2)
            spec1 = torch.nn.functional.pad(spec1, (0, padding_size))
        return spec1, spec2

    def calculate_cosine_similarity(self, spec1, spec2):
        # Flatten the spectrograms and convert them to NumPy arrays for the cosine similarity calculation
        spec1_flat = spec1.numpy().flatten()
        spec2_flat = spec2.numpy().flatten()
        # Calculate the cosine similarity and ensure it is within the range [0, 1]
        sim = 1 - cosine(spec1_flat, spec2_flat)
        # Inform the user if the initial similarity value is negative
        if sim < 0:
            bt.logging.error(f"Initial cosine similarity was negative: {sim}. Setting to 0.")
            sim = 0  # Zero out negative value
        return sim

    def compare_audio(self, file_path1, file_path2, input_text):
        # Extract Mel Spectrograms
        try:
            bt.logging.info(f"Extracting Mel spectrograms...")
            bt.logging.info(f"File 1: {file_path1}")
            bt.logging.info(f"File 2: {file_path2}")
            bt.logging.info(f"Input Text:{input_text}")
            spec1 = self.extract_mel_spectrogram(file_path1)
            spec2 = self.extract_mel_spectrogram(file_path2)
        except Exception as e:
            bt.logging.error(f"Error extracting Mel spectrograms: {e}")
            spec1 = spec2 = None

        if spec1 is not None and spec2 is not None:
            # Pad or Trim
            spec1, spec2 = self.pad_or_trim_to_same_length(spec1, spec2)
            # Calculate Cosine Similarity
            cosine_sim = self.calculate_cosine_similarity(spec1, spec2)
            bt.logging.info(f"Cosine Similarity for Voice Cloning: {cosine_sim}")
        else:
            cosine_sim = 0  # Assigning a default low value if spectrograms extraction failed

        try:
            nisqa_wer_score = score(file_path2, input_text)
        except Exception as e:
            bt.logging.error(f"Error calculating NISQA score inside compare_audio function: {e}")
            nisqa_wer_score = 0

        # Calculate Final Score with 40% weight for Cosine Similarity and 60% weight for NISQA score
        final_score = 0.4 * cosine_sim + 0.6 * nisqa_wer_score
        if cosine_sim == 0:
            final_score = -0.1  # Assigning a negative score for zero or negative cosine similarity
        # bt.logging.info(f"Final Score for Voice Cloning: {final_score}")
        bt.logging.info(f"Final Score: {final_score} and Cosine Similarity: {cosine_sim} for Voice Cloning: ")


        return final_score