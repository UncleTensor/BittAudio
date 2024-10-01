import bittensor as bt
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field


class MusicGeneration(bt.Synapse, BaseModel):
    """
    A class that transforms textual descriptions into music using machine learning models such as 
    'facebook/musicgen-medium' and 'facebook/musicgen-large'. Extends bt.Synapse for seamless integration into a 
    broader neural-based generative system.
    """
    class Config:
        """ Configuration for validation on attribute assignment and strict data handling. """
        validate_assignment = True
        protected_namespaces = ()

    text_input: str = Field(
        ...,
        title="Text Input",
        description="Textual directives or descriptions intended to guide the music generation process."
    )
    model_name: Optional[Literal['facebook/musicgen-medium', 'facebook/musicgen-large']] = Field(
        'facebook/musicgen-medium',
        title="Model Name",
        description="The machine learning model employed for music generation. Supported models: "
                    "'facebook/musicgen-medium', 'facebook/musicgen-large'."
    )
    music_output: Optional[List[Any]] = Field(
        default=None,
        title="Music Output",
        description="The resultant music data, encoded as a list of bytes, generated from the text input."
    )
    
    duration: Optional[int] = Field(
        default=None,
        title="Duration",
        description="The length of the generated music piece, specified in seconds."
    )


    def deserialize(self) -> List:
        """
        Processes and returns the music_output into a format ready for audio rendering or further analysis.
        """
        return self
    