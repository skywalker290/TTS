import os
import pandas as pd
import tensorflow as tf
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.configs import FastSpeechConfig
from tensorflow_tts.models import TFFastSpeech
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.datasets.abstract_dataset import AbstractDataset

class HindiDataset(AbstractDataset):
    def __init__(self, metadata_filename, processor, max_seq_length=200, mel_length_threshold=0, n_files=None):
        self.processor = processor
        self.metadata = pd.read_csv(metadata_filename)
        self.max_seq_length = max_seq_length
        self.mel_length_threshold = mel_length_threshold
        if n_files:
            self.metadata = self.metadata[:n_files]

    def get_all_data(self):
        return self.metadata

    def parse_text(self, text):
        return self.processor.text_to_sequence(text)

    def get_example(self, i):
        sample = self.metadata.iloc[i]
        text = sample['transcript']
        wav_path = sample['wav_path']

        # Load and preprocess wav file
        wav = self.processor.load_wav(wav_path)
        mel = self.processor.melspectrogram(wav)

        # Preprocess text
        text_ids = self.parse_text(text)

        return {
            "utt_id": i,
            "input_ids": text_ids,
            "input_lengths": len(text_ids),
            "mel_gts": mel,
            "mel_lengths": mel.shape[0]
        }

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        return self.get_example(idx)

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
train_dataset = HindiDataset(
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
