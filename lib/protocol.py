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

from typing import Optional, List
import bittensor as bt

class TextToSpeech(bt.Synapse):
    """
    TextToSpeech class inherits from bt.Synapse.
    It is used to convert text to speech.
    """
    # Required request input, filled by sending dendrite caller.
    text_input: Optional[str] = None
    model_name: Optional[str] = None

    # Here we define speech_output as an Optional PyTorch tensor instead of bytes.
    speech_output: Optional[List] = None

    completion: str = None


    def deserialize(self) -> List:
        """
        Deserialize the speech_output into a PyTorch tensor.
        """
        # If speech_output is a tensor, just return it
        # if isinstance(self.speech_output, List):
          # print(" Deserialize the speech_output into a PyTorch tensor.",self)
        return self
        # raise TypeError("speech_output is not a tensor")


class SpeechToText(bt.Synapse):
    """
    SpeechToText class inherits from bt.Synapse.
    It is used to convert speech to text.
    """
    # Required request input, filled by sending dendrite caller.
    speech_input: Optional[List] = None

    # Here we define text_output as an Optional PyTorch tensor instead of bytes.
    text_output: Optional[str] = None

    completion: str = None


    def deserialize(self) -> str:
        """
        Deserialize the text_output into a PyTorch tensor.
        """
        # If text_output is a tensor, just return it
        if isinstance(self.text_output, str):
          print(" Deserialize the text_output into a PyTorch tensor.",self)
          return self
        raise TypeError("text_output is not a tensor")
    

class TextToMusic(bt.Synapse):
    """
    TextToMusic class inherits from bt.Synapse.
    It is used to convert text to music.
    """
    # Required request input, filled by sending dendrite caller.
    text_input: Optional[str] = None

    # Here we define music_output as an Optional PyTorch tensor instead of bytes.
    music_output: Optional[List] = None

    completion: str = None


    def deserialize(self) -> List:
        """
        Deserialize the music_output into a PyTorch tensor.
        """
        # If music_output is a tensor, just return it
        if isinstance(self.music_output, List):
          print(" Deserialize the music_output into a PyTorch tensor.",self)
          return self
        raise TypeError("music_output is not a tensor")
    
class TextToSound(bt.Synapse):
    """
    TextToSound class inherits from bt.Synapse.
    It is used to convert text to sound.
    """
    # Required request input, filled by sending dendrite caller.
    text_input: Optional[str] = None

    # Here we define sound_output as an Optional PyTorch tensor instead of bytes.
    sound_output: Optional[List] = None

    completion: str = None


    def deserialize(self) -> List:
        """
        Deserialize the sound_output into a PyTorch tensor.
        """
        # If sound_output is a tensor, just return it
        if isinstance(self.sound_output, List):
          print(" Deserialize the sound_output into a PyTorch tensor.",self)
          return self
        raise TypeError("sound_output is not a tensor")
    

class VoiceClone(bt.Synapse):
    """
    VoiceClone class inherits from bt.Synapse.
    It is used to clone a voice.
    """
    text_input: Optional[str] = None

    # Required request input, filled by sending dendrite caller.
    clone_input: Optional[List] = None

    # Here we define speech_output as an Optional PyTorch tensor instead of bytes.
    clone_output: Optional[List] = None

    completion: str = None


    def deserialize(self) -> List:
        """
        Deserialize the speech_output into a PyTorch tensor.
        """
        # If speech_output is a tensor, just return it
        if isinstance(self.speech_output, List):
          print(" Deserialize the speech_output into a PyTorch tensor.",self)
          return self
        raise TypeError("speech_output is not a tensor")