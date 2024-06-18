import os
import pandas as pd

# Define paths
base_dir = '/content/Hindi_M'
wav_dir = os.path.join(base_dir, 'wav')
txt_dir = os.path.join(base_dir, 'txt')

# Create a list to store data
data = []

# Iterate through the wav directory
for wav_file in os.listdir(wav_dir):
    if wav_file.endswith('.wav'):
        # Get the ID of the wav file
        file_id = wav_file.replace('.wav', '')
        wav_path = os.path.join(wav_dir, wav_file)
        txt_path = os.path.join(txt_dir, f'{file_id}.txt')
        
        # Read the transcript
        with open(txt_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        
        # Append to the data list
        data.append([wav_path, transcript])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['wav_path', 'transcript'])

# Save to CSV
csv_path = os.path.join(base_dir, 'metadata.csv')
df.to_csv(csv_path, index=False)
