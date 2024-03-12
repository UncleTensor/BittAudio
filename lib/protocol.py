# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
# Copyright © 2023 ETG

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from typing import List, Optional
import bittensor as bt
from pydantic import Field

class TextToSpeech(bt.Synapse):
    """
    This class converts text into speech using predefined machine learning models. Available models include 'microsoft/speecht5_tts', 'facebook/mms-tts-eng', 'suno/bark', and 'elevenlabs/eleven'. Inherits functionalities from bt.Synapse, making it integral to a neural processing architecture.
    """
    class Config:
        """
        Configuration class for Pydantic model, enforcing strict type validation and assignment, thereby ensuring data integrity for TextToSpeech instances.
        """
        validate_assignment = True

    text_input: str = Field(
        default=None,
        title="Text Input",
        description="The input text to be converted into speech format."
    )
    model_name: Optional[str] = Field(
        default=None,
        title="Model Name",
        description="Specifies the machine learning model used for text-to-speech conversion. Supported models: 'microsoft/speecht5_tts', 'facebook/mms-tts-eng', 'suno/bark', 'elevenlabs/eleven'."
    )
    clone_input: List = Field(
        default=None,
        title="Clone Input",
        description="A list of parameters used for enhancing the text-to-speech process, relevant for models supporting voice cloning."
    )
    speech_output: List = Field(
        default=None,
        title="Speech Output",
        description="The resulting speech data produced from the text input."
    )

    def deserialize(self) -> List:
        """
        Converts and returns the speech_output into a structured format, suitable for playback or further processing.
        """
        return self

class MusicGeneration(bt.Synapse):
    """
    A class that transforms textual descriptions into music using machine learning models such as 'facebook/musicgen-medium' and 'facebook/musicgen-large'. Extends bt.Synapse, facilitating its integration into a broader neural-based generative system.
    """
    class Config:
        """
        Configuration class that mandates validation on attribute assignments, ensuring correctness and reliability of data for MusicGeneration instances.
        """
        validate_assignment = True

    text_input: str = Field(
        default=None,
        title="Text Input",
        description="Textual directives or descriptions intended to guide the music generation process."
    )
    model_name: Optional[str] = Field(
        default=None,
        title="Model Name",
        description="The machine learning model employed for music generation. Supported models: 'facebook/musicgen-medium', 'facebook/musicgen-large'."
    )
    music_output: List = Field(
        default=None,
        title="Music Output",
        description="The resultant music data, encoded as a list, generated from the text input."
    )
    duration: int = Field(
        default=None,
        title="Duration",
        description="The length of the generated music piece, specified in seconds."
    )

    def deserialize(self) -> List:
        """
        Processes and returns the music_output into a format ready for audio rendering or further analysis.
        """
        return self

class VoiceClone(bt.Synapse):
    """
    The VoiceClone class is aimed at replicating specific voice attributes for text-to-speech applications using models like 'suno/bark' and 'elevenlabs/eleven'. It operates within the framework provided by bt.Synapse to harness advanced neural synthesis techniques.
    """
    class Config:
        """
        A configuration class that enables strict validation rules for property assignments, maintaining the VoiceClone class's output accuracy and consistency.
        """
        validate_assignment = True

    text_input: str = Field(
        default=None,
        title="Text Input",
        description="Text content to be synthesized using cloned voice attributes."
    )
    clone_input: List = Field(
        default=None,
        title="Clone Input",
        description="Data used for analyzing and replicating the desired voice characteristics, an audio array constructed from mp3 or wav files."
    )
    clone_output: List = Field(
        default=None,
        title="Clone Output",
        description="The synthesized voice output, incorporating cloned attributes, delivered as a list."
    )
    sample_rate: int = Field(
        default=None,
        title="Sample Rate",
        description="The sample rate of the output audio, integral for determining audio quality and clarity."
    )
    hf_voice_id: str = Field(
        default=None,
        title="Hugging Face Voice ID",
        description="An identifier for the voice model used from Hugging Face's repository. Supported models for cloning include 'suno/bark' and 'elevenlabs/eleven'."
    )
    model_name: Optional[str] = Field(
        default=None,
        title="Model Name",
        description="The name of the machine learning model used for voice cloning. Compatible models: 'suno/bark', 'elevenlabs/eleven'."
    )

    def deserialize(self) -> "VoiceClone":
        """
        Processes and returns the clone_output for use in auditory playback or further analysis.
        """
        return self

