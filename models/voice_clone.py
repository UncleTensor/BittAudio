from elevenlabs import clone, generate
from elevenlabs import set_api_key


class ElevenLabsClone():
    def __init__(self, api_key):
        self.api_key = api_key


    def clone_voice(self,text_input,source_file, hf_voice_id):
        try:
            set_api_key(self.api_key)

            voice = clone(
                name= hf_voice_id,
                description="any", # Optional
                files=[source_file],
                model="eleven_multilingual_v2"
            )

            audio = generate(text= text_input, voice=voice)

            return audio
        except Exception as e:
            print(f"An error occurred: {e}")
    
