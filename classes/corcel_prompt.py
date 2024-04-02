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
    
    def generate_prompt(self):
        character_archetypes = [
            "The Wanderer", "The Hero", "The Fool", "The Sage", "The Villain", "The Orphan", "The Magician", 
            "The Outlaw", "The Explorer", "The Rebel", "The Lover", "The Dreamer", "The King", "The Queen", 
            "The Jester", "The Priest", "The Pirate", "The Knight", "The Princess", "The Prince", "The Witch", 
            "The Wizard", "The Guardian", "The Warrior", "The Healer", "The Inventor", "The Mentor", "The Joker", 
            "The Scientist", "The Philosopher", "The Artist", "The Spy", "The Detective", "The Mercenary", 
            "The Gladiator", "The Seer", "The Prophet", "The Shaman", "The Bard", "The Ranger", "The Assassin", 
            "The Monk", "The Paladin", "The Alchemist", "The Scribe", "The Oracle", "The Necromancer", 
            "The Diplomat", "The Beast", "The Changeling", "The Summoner", "The Thief", "The Berserker"
        ]

        settings = [
            "Overgrown jungle ruins", "Abandoned space station", "Desolate desert city", "Hidden underwater kingdom", 
            "Ancient crumbling castle", "Futuristic dystopian city", "Remote alien planet", "Enchanted forest", 
            "Isolated mountain monastery", "Underground network of caves", "Labyrinthine library", "Mysterious ghost town", 
            "Secluded island paradise", "Opulent royal court", "War-torn battlefield", "Creepy haunted house", 
            "Busy metropolitan streets", "Tranquil rural village", "High-tech space colony", "Dense and dark forest", 
            "Snow-covered arctic base", "Luxurious cruise ship", "Time-traveling to historical events", "Virtual reality world", 
            "Post-apocalyptic wasteland", "Interdimensional crossroads", "Secret society headquarters", "Ancient pyramid complex", 
            "Giant skyscraper city", "Lost continent of Atlantis", "Deep-sea alien base", "High stakes casino", 
            "Mad scientist's laboratory", "A parallel universe", "Majestic waterfall city", "Gloomy Victorian London", 
            "Sun-soaked Caribbean island", "Cyberpunk Tokyo", "Ancient Roman Colosseum", "Middle Eastern bazaar", 
            "Old West frontier town", "Spectral otherworld", "Steampunk airship", "Prehistoric jungle", "Fairy-tale castle in the clouds", 
            "Mythical Mount Olympus", "Norse village under the Northern Lights", "Forbidden Eastern temple", "Hidden valley of Shangri-La"
        ]

        themes = [
            "Discovery", "Revenge", "Love", "Betrayal", "Friendship", "Courage", "Fear", "Greed", "Honor", "Justice", 
            "Loss", "Memory", "Freedom", "Chaos", "Duty", "Hope", "Despair", "Power", "Truth", "Deception", 
            "Survival", "Sacrifice", "Redemption", "Curiosity", "Death", "Faith", "Guilt", "Innocence", "Jealousy", "Ambition", 
            "Compassion", "Forgiveness", "Hatred", "Identity", "Knowledge", "Loneliness", "Passion", "Rebirth", "Strength", "Weakness", 
            "Adventure", "Beauty", "Creation", "Destruction", "Enlightenment", "Fate", "Glory", "Heritage", "Illusion", "Justice", 
            "Kinship", "Legacy", "Mystery", "Nobility", "Obsession", "Pride", "Quest", "Rivalry", "Spirituality", "Transformation"
        ]

        emotions = [
            "Curiosity", "Fear", "Joy", "Despair", "Hope", "Anger", "Love", "Sadness", "Excitement", "Nostalgia", 
            "Anxiety", "Confidence", "Embarrassment", "Gratitude", "Loneliness", "Shock", "Euphoria", "Frustration", "Pride", "Shame", 
            "Contentment", "Disgust", "Envy", "Guilt", "Happiness", "Melancholy", "Passion", "Regret", "Surprise", "Wistfulness", 
            "Ambivalence", "Boredom", "Compassion", "Determination", "Empathy", "Forgiveness", "Greed", "Humility", "Inspiration", "Jealousy", 
            "Kindness", "Lust", "Optimism", "Pessimism", "Relief", "Sorrow", "Trust", "Vulnerability", "Warmth", "Zeal", "Awe", 
            "Contempt", "Delight", "Eagerness", "Fearlessness", "Generosity", "Hopelessness", "Indignation", "Joyfulness", "Mirth", "Resignation"
        ]

        objects = [
            "Ancient tome", "Mysterious orb", "Forgotten artifact", "Enchanted sword", "Magical amulet", "Hidden treasure", "Secret diary", 
            "Alien device", "Cursed gem", "Legendary shield", "Lost manuscript", "Mystical ring", "Sacred relic", "Time machine", "Otherworldly key", 
            "Phantom mirror", "Ancient scroll", "Elixir of life", "Invisible cloak", "Celestial map", "Dimensional gateway", "Eternal flame", 
            "Golden compass", "Haunted locket", "Infernal book", "Jeweled scepter", "Kaleidoscopic crystal", "Lunar pendant", "Mythical harp", 
            "Necromancer's staff", "Omniscient orb", "Pandora's box", "Quantum computer", "Runestone", "Sorcerer's stone", "Transmutation tablet", 
            "Undying lantern", "Vortex manipulator", "Wishing well", "Xenobotanical seed", "Yggdrasil leaf", "Zodiac talisman", "Arcane grimoire", 
            "Bewitched vial", "Cryptic codex", "Draconic egg", "Empyrean horn", "Feywild blossom", "Grimoire of shadows", "Hellfire torch", 
            "Icefire opal", "Jade dragon statue", "Kraken's tooth", "Lich's phylactery", "Moonstone", "Netherworld portal", "Obsidian dagger"
        ]

        action_verbs = [
            "Navigate", "Explore", "Uncover", "Challenge", "Defeat", "Discover", "Embrace", "Fight", "Guard", "Heal", 
            "Invent", "Journey", "Know", "Learn", "Master", "Notice", "Observe", "Pursue", "Question", "Reveal", 
            "Solve", "Transform", "Understand", "Venture", "Win", "X-ray", "Yearn", "Zoom", "Adapt", "Build", 
            "Create", "Design", "Enlighten", "Forge", "Generate", "Harmonize", "Inspire", "Join", "Kindle", "Link", 
            "Manifest", "Nurture", "Organize", "Prepare", "Quell", "Rescue", "Sustain", "Teach", "Unite", "Validate", 
            "Wander", "Xenialize", "Yield", "Zest", "Affirm", "Blossom", "Cultivate", "Delve", "Elevate", "Facilitate", "Glow", "Honor", "Illuminate"
        ]

        sensory_details = [
            "rustling leaves", "whispering winds", "echoing footsteps", "shimmering lights", "crackling fires", "howling wolves", "gurgling streams", 
            "chirping crickets", "buzzing bees", "singing birds", "roaring thunder", "flashing lightning", "pattering rain", "blowing snow", 
            "glowing moonlight", "bright sunshine", "fading twilight", "rising dawn", "falling dusk", "blooming flowers", "waving grass", "flowing rivers", 
            "crashing waves", "salty sea air", "fresh mountain breeze", "humid jungle mist", "dry desert heat", "cold arctic chill", "warm tropical sun", 
            "cool forest shade", "damp cave air", "scent of pine", "aroma of spices", "fragrance of perfume", "smell of rain", "taste of salt", 
            "flavor of sweet", "touch of silk", "texture of sand", "sound of silence", "sight of stars", "vision of the future", "feeling of anticipation", 
            "sensation of freedom", "experience of awe", "moment of clarity", "whiff of coffee", "glimpse of the unknown", "echo of the past"
        ]

        cultural_references = [
            "Aztec civilization", "Roman Empire", "Ancient Greece", "Medieval Europe", "Renaissance art", "Victorian England", "Edo period Japan", 
            "Ancient Egypt", "Ottoman Empire", "Mongol invasions", "Silk Road travels", "Age of Exploration", "French Revolution", "American Wild West", 
            "Roaring Twenties", "Cold War espionage", "Viking raids", "Pirate legends", "Chinese dynasties", "Russian folklore", "Greek mythology", 
            "Norse mythology", "Egyptian gods", "Celtic legends", "Native American traditions", "African tribal heritage", "South American tribes", 
            "Australian aboriginal cultures", "Polynesian navigation", "Caribbean piracy", "Indian Vedas", "Middle Eastern bazaars", "European fairy tales", 
            "Arabian Nights", "Japanese samurai", "Korean Joseon dynasty", "Persian poetry", "Mayan astronomy", "Incan architecture", "Hawaiian mythology", 
            "Maori warriors", "Scandinavian fjords", "Amazonian rainforests", "Saharan caravans", "Alpine folklore", "Balkan history", "Silicon Valley innovators", 
            "Hollywood golden age", "Broadway musicals", "Harlem Renaissance", "British Invasion music", "1960s counterculture", "Ancient Sumerians", "Byzantine Empire"
        ]

        base_prompt = "Imagine a [Character Archetype] in a [Setting] on an adventure driven by [Theme] and [Emotion]. They find a [Object] that unfolds mysteries. Detail their journey with an [Action Verb] and [Sensory Detail]. Include a [Cultural Reference] for depth. What profound discovery shapes their experience? Craft a sentence under 256 characters."

        prompt = base_prompt.replace("[Character Archetype]", random.choice(character_archetypes))
        prompt = prompt.replace("[Setting]", random.choice(settings))
        prompt = prompt.replace("[Theme]", random.choice(themes))
        prompt = prompt.replace("[Emotion]", random.choice(emotions))
        prompt = prompt.replace("[Object]", random.choice(objects))
        prompt = prompt.replace("[Action Verb]", random.choice(action_verbs))
        prompt = prompt.replace("[Sensory Detail]", random.choice(sensory_details))
        prompt = prompt.replace("[Cultural Reference]", random.choice(cultural_references))
        return prompt
    
    def filter_prompt(self, text):
        """
        Returns the first 250 characters of the given text, without truncating a word.
        """
        # Find the last space character within the first 250 characters.
        last_space_index = text[:250].rfind(' ')

        # If there is no space character within the first 250 characters, return the first 250 characters.
        if last_space_index == -1:
            return text[:250]

        # Otherwise, return the substring up to the last space character.
        return text[:last_space_index]

    def get_TTS(self):
        seed_prompt = self.generate_prompt()
        data = {
            "messages": [{"role": "user", "content": seed_prompt }],
            "miners_to_query": 1,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        prompt_to_filter = self.filter_prompt(self.post_request(data))
        return prompt_to_filter
    
    def get_VC(self):
        seed_prompt = self.generate_prompt()
        data = {
            "messages": [{"role": "user", "content": seed_prompt }],
            "miners_to_query": 1,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        prompt_to_filter = self.filter_prompt(self.post_request(data))
        return prompt_to_filter
    
    def get_TTM(self):
        prompt = self.generate_music_prompt()
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "miners_to_query": 1,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        return self.post_request(data)