import pandas as pd
import subprocess
import time
import os
import re

def extract_column_positions(header_line):
    # Extract the start and end positions of each header
    columns = ["UID","NAME", "ADDRESS" "STAKE(τ)", "SENATOR"]
    positions = []
    for col in columns:
        start = header_line.find(col)
        end = start + len(col)
        positions.append((start, end))
    return positions

def fetch_metagraph_data():
    while True:
        try:
            output = subprocess.check_output(['btcli', 'r', 'list'])
            lines = output.split("\n")

            header_positions = []
            data = []
            
            # Identify header and extract column positions
            for line in lines:
                if line.startswith("UID"):
                    header_positions = extract_column_positions(line)
                    break

            # Extract data based on header positions
            for line in lines:
                if line and line[0].isdigit():
                    row = [line[start:end].strip() for start, end in header_positions]
                    data.append(row)

            # Convert the extracted data to DataFrame and save it in CSV format
            df = pd.DataFrame(data, columns=["UID","NAME", "ADDRESS" "STAKE(τ)", "SENATOR"])
            df.to_csv(f'static/rlist.csv', index=False)

            break
        except subprocess.CalledProcessError:
            time.sleep(1)

fetch_metagraph_data()