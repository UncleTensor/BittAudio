import requests
import bittensor as bt
import os
import random

class CorcelAPI:
    def __init__(self):
        self.base_url = "https://api.corcel.io/v1/text/cortext/chat"
        self.api_key = os.environ.get('CORCEL_API_KEY')
        if self.api_key is None:
            bt.logging.info(f"Corcel API is: {self.api_key}")
            bt.logging.error("Corcel API key not found.")
            pass
        self.headers = {
            "Authorization": self.api_key,
            "accept": "application/json",
            "content-type": "application/json"
        }
    
    def post_request(self, data):
        response = requests.post(self.base_url, headers=self.headers, json=data)
        # Check the HTTP status code
        if response.status_code == 200:
            # If status code is 200, parse the response
            json_data = response.json()
            content = json_data[0]['choices'][0]['delta']['content']
            return content
        else:
            # Handle other failure codes
            return None

    def generate_music_prompt(self):
        genres = ["rock", "alternative rock", "classic rock", "punk rock", "hard rock", "indie rock", "progressive rock", "pop", "teen pop", "pop rock",
                  "glo-fi", "hyperpop", "progressive trance", "psybient", "dark ambient", "space music", "ambient techno", "ebm (electronic body music)", 
                  "electro pop", "indie pop", "k-pop", "jazz", "smooth jazz", "bebop", "swing", "fusion", "dixieland", "free jazz", "classical", "baroque", 
                  "singer/songwriter", "reggae", "dancehall", "dub", "rocksteady", "ska", "blues", "delta blues", "chicago blues", "electric blues", "metal", 
                  "romantic", "modern classical", "opera", "chamber music", "electronic", "house", "techno", "dubstep", "trance", "drum and bass", "ambient", 
                  "hip hop/rap", "gangsta rap", "trap", "old school", "crunk", "cloud rap", "r&b/soul", "contemporary r&b", "soul", "funk", "motown", "neo-soul", 
                  "sufi music", "bolero", "fado", "zouk", "afro-cuban jazz", "nu-jazz", "hard bop", "cool jazz", "modal jazz", "sufi rock", "hindustani classical", 
                  "country", "classic country", "pop country", "bluegrass", "honky tonk", "americana", "folk", "traditional folk", "contemporary folk", "folk rock", 
                  "freestyle", "crust punk", "digital hardcore", "jangle pop", "riot grrrl", "neo-psychedelia", "acid rock", "stoner rock", "doom metal", "sludge metal", 
                  "heavy metal", "death metal", "black metal", "thrash metal", "power metal", "latin", "salsa", "reggaeton", "bachata", "tango", "merengue", "world music", 
                  "african", "caribbean", "celtic", "middle eastern", "electronic dance music (edm)", "house", "techno", "trance", "drum and bass", "dubstep", "math rock", 
                  "post-punk", "new wave", "grunge", "shoegaze", "lo-fi", "highlife", "soukous", "bhangra", "gypsy jazz", "vaporwave", "future bass", "hardstyle", "psytrance", 
                  "chiptune", "folk metal", "viking metal", "post-metal", "grime", "uk garage", "2-step", "visual kei", "j-rock", "c-pop", "mandopop", "qawwali", "ghazal", 
                  "carnatic classical", "gregorian chant", "noise music", "industrial rock", "gothic rock", "ethereal wave", "darkwave", "bubblegum pop", "disco", "eurodance", 
                  "industrial metal", "nu metal", "rap rock", "rap metal", "horrorcore", "electroclash", "new rave", "minimalism", "contemporary folk music", "progressive bluegrass", 
                  "electroacoustic music", "drone music", "witch house", "future garage", "jersey club", "baile funk", "kawaii metal", "nintendocore", "vapor trap", "chillwave", 
                  "idm (intelligent dance music)", "breakcore", "witch house"]        
        genre_combinations = ["indie folk-pop", "electro funk-house", "progressive metal-jazz fusion", "reggae-dancehall-hip hop", "neo-soul-r&b", 
                              "ska-punk-rocksteady", "trance-techno-ambient", "bluegrass-country-folk", "afrobeat-caribbean funk", "chillhop-lo-fi jazz", 
                              "smooth jazz-electro pop", "classic rock-blues", "drum and bass-dubstep", "ambient-world music", "pop country-americana", 
                              "latin-salsa-merengue", "heavy metal-hard rock", "singer/songwriter-indie rock", "electro pop-k-pop", "funk-soul-motown", 
                              "house-techno-trance", "bebop-swing-jazz", "alternative rock-punk rock", "reggaeton-latin pop", "celtic-folk-traditional folk", 
                              "psychedelic rock-progressive rock", "ambient-electronic-downtempo", "folk-rock-americana", "jazz fusion-world music", 
                              "synthwave-retro electro", "baroque pop-chamber pop", "flamenco-guitar latin jazz", "trip hop-electro jazz", "classic soul-gospel", 
                              "disco-funk-house", "acid jazz-breakbeat", "dub-roots reggae", "techno-industrial-electro", "opera-classical crossover", 
                              "dark ambient-drone", "minimal techno-deep house", "experimental rock-avant-garde", "gothic metal-symphonic rock", "bossa nova-samba-jazz", 
                              "post-rock-shoegaze", "new age-neo-classical", "glitch-hop-idm", "celtic punk-folk rock", "delta blues-chicago blues", "surf rock-garage rock"]
        emotions = ["excitement", "nostalgia", "joy", "melancholy", "tranquility", "anger", "empowerment", "rebellion", "love", "heartbreak", "hope", "despair", 
                    "reflection", "euphoria", "serenity", "aggression", "contentment", "sorrow", "inspiration", "fear", "loneliness", "bliss", "ennui", "curiosity", 
                    "yearning", "freedom", "resilience", "peace", "wanderlust", "awe", "gratitude", "restlessness", "optimism", "pensive", "grit", "triumph", "solitude", 
                    "anxiousness", "celebration", "mystery", "anticipation", "conflict", "elation", "exhaustion", "intimacy", "invigoration", "melancholia", 
                    "mischievousness", "nervousness", "pride", "regret", "relaxedness", "remorse", "sanguineness", "sensuality", "shock", "suspense", "sympathy", 
                    "tenderness", "terror", "thrill", "vulnerability", "warmth", "whimsy", "wistfulness", "zeal", "zenith", "zest", "adoration", "ambivalence"]
        
        # Randomly select one item from each list
        genre = random.choice(genres)
        genre_combination = random.choice(genre_combinations)
        emotion = random.choice(emotions)

        # Construct the prompt
        prompt = f"Create a unique music prompt using {genre} or {genre_combination} to evoke {emotion}. Keep it under 20 words."

        # Return the constructed prompt
        return prompt
    
    def get_TTS(self):
        data = {
            "messages": [{"role": "user", "content": "Based on all the books, articles, documents or dataset you have, give me a single random and unique different sentence less than 200 characters. It could have a minimum of 200 words, but not more than 220 characters. Use only letters and not numbers in the sentence. If we have numbers, convert it to letters. Sentence should not have more than 10% of the words in common with each other."}],
            "miners_to_query": 3,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        return self.post_request(data)
    
    def get_VC(self):
        data = {
            "messages": [{"role": "user", "content": "Based on all the books, articles, documents or dataset you have, give me a single random and unique different sentence less than 200 characters. It could have a minimum of 200 words, but not more than 220 characters. Use only letters and not numbers in the sentence. If we have numbers, convert it to letters. Sentence should not have more than 10% of the words in common with each other."}],
            "miners_to_query": 3,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        return self.post_request(data)
    
    def get_TTM(self):
        prompt = self.generate_music_prompt()
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "miners_to_query": 3,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        return self.post_request(data)