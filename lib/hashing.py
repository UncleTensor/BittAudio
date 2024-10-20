import hashlib
import os
import json
from datetime import datetime
import bittensor as bt

hashes_file = 'music_hashes.json'

# In-memory cache for fast lookup
cache = set()

def calculate_audio_hash(audio_data: bytes) -> str:
    """Calculate a SHA256 hash of the given audio data."""
    return hashlib.sha256(audio_data).hexdigest()

def load_hashes_to_cache():
    """Load existing hashes from JSON file into in-memory cache."""
    if os.path.exists(hashes_file):
        with open(hashes_file, 'r') as file:
            data = json.load(file)
            for entry in data:
                cache.add(entry['hash'])  # Add hash to in-memory cache

def save_hash_to_file(hash_value: str, timestamp: str,  miner_id: str = None):
    """Save the new hash to the JSON file and in-memory cache."""
    cache.add(hash_value)  # Add to cache for fast lookups
    if os.path.exists(hashes_file):
        with open(hashes_file, 'r+') as file:
            data = json.load(file)
            data.append({'hash': hash_value, 'miner_id': miner_id, 'timestamp': timestamp})
            file.seek(0)
            json.dump(data, file)
    else:
        # If the file doesn't exist, create it with the initial hash entry
        with open(hashes_file, 'w') as file:
            json.dump([{'hash': hash_value, 'miner_id': miner_id, 'timestamp': timestamp}], file)
            

def check_duplicate_music(hash_value: str) -> bool:
    """Check if the given hash already exists in the in-memory cache."""
    return hash_value in cache

def process_miner_music(miner_id: str, audio_data: bytes):
    """Process music sent by a miner and check for duplicates."""
    audio_hash = calculate_audio_hash(audio_data)  # Calculate the audio hash

    if check_duplicate_music(audio_hash):  # Check if it's a duplicate
        bt.logging.info(f"Duplicate music detected from miner: {miner_id}")
        return # Do nothing if it's a duplicate
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_hash_to_file(audio_hash, miner_id, timestamp)  # Save the hash to the file and cache
        bt.logging.info(f"Music processed and saved successfully for miner: {miner_id}")
        return audio_hash
