import os
import pandas as pd
import tensorflow_tts
import tensorflow as tf
from tensorflow_tts.inference import AutoProcessor, TFAutoModel
from tensorflow_tts.configs import FastSpeechConfig
from tensorflow_tts.models import TFFastSpeech
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.datasets import CharactorDataset

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

# Define the path to save the trained model
model_save_path = './fastspeech_model_hindi'

# Load the processor
processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")

# Load the dataset
train_dataset = CharactorDataset(
    metadata_filename=csv_path,
    processor=processor,
    max_seq_length=200,
    mel_length_threshold=0,
    n_files=None,
)

# Define the FastSpeech model configuration
config = FastSpeechConfig()

# Initialize the model
fastspeech = TFFastSpeech(config=config, name="fastspeech")

# Define the trainer
trainer = Seq2SeqBasedTrainer(
    config=config,
    model=fastspeech,
    train_dataset=train_dataset,
    eval_dataset=None,  # Add evaluation dataset if available
)

# Train the model
trainer.fit()

# Save the model
fastspeech.save(model_save_path)
