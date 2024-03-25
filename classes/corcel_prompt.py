import requests
import bittensor as bt
import os

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
        data = {
            "messages": [{"role": "user", "content": "random Music generation phrase for AI music generation model in less than 32 words"}],
            "miners_to_query": 3,
            "top_k_miners_to_query": 40,
            "ensure_responses": True,
            "model": "cortext-ultra",
            "stream": False
        }
        return self.post_request(data)