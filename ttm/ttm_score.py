import os
import torch
import torchaudio
import numpy as np
import bittensor as bt
from sre_constants import error
from huggingface_hub import hf_hub_download
from audiocraft.metrics import PasstKLDivergenceMetric
from audiocraft.metrics import CLAPTextConsistencyMetric
from audioldm_eval.metrics.fad import FrechetAudioDistance



class MetricEvaluator:
    @staticmethod
    def calculate_kld(generated_audio_dir, target_audio_dir):
        try:
            # Get the single audio file path in the directory
            generate = next((f for f in os.listdir(generated_audio_dir) if os.path.isfile(os.path.join(generated_audio_dir, f))), None)
            target = next((f for f in os.listdir(target_audio_dir) if os.path.isfile(os.path.join(target_audio_dir, f))), None)

            if generate is None or target is None:
                bt.logging.error("Generated or target audio file not found.")
                return None

            # Load your predicted and target audio files
            target_waveform, target_sr = torchaudio.load(os.path.join(target_audio_dir, target))
            generated_waveform, generated_sr = torchaudio.load(os.path.join(generated_audio_dir, generate))

            # Ensure sample rates match
            if target_sr != generated_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=generated_sr, new_freq=target_sr)
                generated_waveform = resampler(generated_waveform)
                generated_sr = target_sr

            # Truncate or pad waveforms to match lengths
            min_length = min(target_waveform.shape[-1], generated_waveform.shape[-1])
            target_waveform = target_waveform[..., :min_length]
            generated_waveform = generated_waveform[..., :min_length]

            # Ensure that the audio tensors are in the shape [batch_size, channels, length]
            target_waveform = target_waveform.unsqueeze(0)  # Adding batch dimension
            generated_waveform = generated_waveform.unsqueeze(0)  # Adding batch dimension

            # The sizes of the waveform
            sizes = torch.tensor([target_waveform.shape[-1]])

            # The sample rates
            sample_rates = torch.tensor([target_sr])  # Use just one sample rate as they should match

            # Initialize the PasstKLDivergenceMetric
            kld_metric = PasstKLDivergenceMetric()

            # Move tensors to the appropriate device if needed
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            target_waveform = target_waveform.to(device)
            generated_waveform = generated_waveform.to(device)
            sizes = sizes.to(device)
            sample_rates = sample_rates.to(device)
            kld_metric = kld_metric.to(device)

            # Update the metric
            kld_metric.update(preds=generated_waveform, targets=target_waveform, sizes=sizes, sample_rates=sample_rates)

            # Compute the PasstKLDivergenceMetric score
            kld = kld_metric.compute()
            return kld['kld_both']

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            bt.logging.error(f"Error during KLD calculation: {e}\n{traceback_str}")
            return None


    @staticmethod
    def calculate_fad(generated_audio_dir, target_audio_dir):
      # Initialize the Frechet Audio Distance calculator
      fad_calculator = FrechetAudioDistance()

      # Calculate the FAD score between the two directories
      fad_score = fad_calculator.score(
          background_dir=generated_audio_dir,  # Generated audio directory
          eval_dir=target_audio_dir,           # Target audio directory
          store_embds=False,                   # Set to True if you want to store embeddings for later reuse
          limit_num=1,                      # Limit the number of files to process, None means no limit
          recalculate=True                     # Set to True if you want to recalculate embeddings
      )

      # Extract the FAD score from the dictionary
      fad_value = fad_score['frechet_audio_distance']

      # Clamp the value to 0 if it's negative
      fad = max(0, fad_value)
      return fad

    @staticmethod
    def calculate_consistency(generated_audio_dir, text):
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

            # Get the single audio file path in the directory
            file_name = next((f for f in os.listdir(generated_audio_dir) if os.path.isfile(os.path.join(generated_audio_dir, f))), None)
            if file_name is None:
                raise FileNotFoundError("No audio file found in the directory.")

            file_path = os.path.join(generated_audio_dir, file_name)

            # Load and process the audio
            audio, sr = torchaudio.load(file_path)
            audio = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=1)

            # Calculate consistency score
            clap_metric.update(audio.unsqueeze(0), [text], torch.tensor([audio.shape[1]]), torch.tensor([sr]))
            consistency_score = clap_metric.compute()

            return consistency_score
        except Exception as e:
            bt.logging.error(f"Error during consistency calculation: {e}")
            return None

class Normalizer:
    @staticmethod
    def normalize_kld(kld_score):
        if kld_score is not None:
            if 0 <= kld_score <= 1:
                normalized_kld = (1 - kld_score)  # Higher score is better, so normalize as 1 - kld_score
            elif 1 < kld_score <= 2:
                normalized_kld = 0.5 * (2 - kld_score)  # Scale down between 0.5 and 0
            else:
                normalized_kld = 0  # Anything > 2 is considered bad
        else:
            normalized_kld = 0
        return normalized_kld

    @staticmethod
    def normalize_fad(fad_score):
        if fad_score is not None:
            if 0 <= fad_score <= 5:
                normalized_fad = (5 - fad_score) / 5  # Normalize between 0 and 1 (higher is better)
            elif 5 < fad_score <= 10:
                normalized_fad = 0.5 * (10 - fad_score) / 5  # Scale down between 0.5 and 0
            else:
                normalized_fad = 0  # Anything > 10 is considered bad
        else:
            normalized_fad = 0
        return normalized_fad


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

    def get_directory(self, path):
        return os.path.dirname(path)

    def evaluate_music_quality(self, generated_audio, target_audio, text=None):
      
      generated_audio_dir = self.get_directory(generated_audio)
      target_audio_dir = self.get_directory(target_audio)

      bt.logging.info(f"Generated audio directory: {generated_audio_dir}")
      bt.logging.info(f"Target audio directory: {target_audio_dir}")

      try:
          kld_score = self.metric_evaluator.calculate_kld(generated_audio_dir, target_audio_dir)
      except:
          bt.logging.error(f"Failed to calculate KLD")

      try:
          fad_score = self.metric_evaluator.calculate_fad(generated_audio_dir, target_audio_dir)
      except:
          bt.logging.error(f"Failed to calculate FAD")

      try:
          consistency_score = self.metric_evaluator.calculate_consistency(generated_audio_dir, text)
      except:
          bt.logging.error(f"Failed to calculate Consistency score")

      # Normalize scores and calculate aggregate score
      normalized_kld = self.normalizer.normalize_kld(kld_score)
      normalized_fad = self.normalizer.normalize_fad(fad_score)

      aggregate_quality = self.aggregator.geometric_mean({'KLD': normalized_kld, 'FAD': normalized_fad})
      aggregate_score = self.aggregator.geometric_mean({'quality': aggregate_quality, 'normalized_consistency': consistency_score}) if consistency_score > 0.1 else 0
        # Print scores in a table
      table1 = [
            ["Metric", "Raw Score"],
            ["KLD Score", kld_score],
            ["FAD Score", fad_score],
            ["Consistency Score", consistency_score]
        ]

      # Print table of normalized scores
      table2 = [
            ["Metric", "Normalized Score"],
            ["Normalized KLD", normalized_kld],
            ["Normalized FAD", normalized_fad],
            ["Consistency Score", consistency_score]
            ]
           
      
      return aggregate_score, table1 , table2