from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer
import torchaudio

class SpeechToTextEvaluator:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device) 
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model.to(self.device)

    def transcribe_audio(self, audio_file):
        # Load the audio file
        waveform, sampling_rate = torchaudio.load(audio_file)

        # Resample if necessary
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            waveform = resampler(waveform)
            sampling_rate = 16000

        # Process the audio input
        inputs = self.processor(waveform.squeeze(), sampling_rate=sampling_rate, return_tensors="pt", padding=True)
#        input_values = inputs.input_values
        input_values = inputs.input_values.to(self.model.device)

        # Ensure the input is 2D [batch_size, sequence_length]
        if input_values.ndim == 3:
            input_values = input_values.squeeze(0)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription


    def evaluate_wer(self, audio_file, reference_text):
        transcription = self.transcribe_audio(audio_file)
        wer_score = wer(reference_text, transcription)
        return wer_score