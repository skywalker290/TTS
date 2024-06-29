# !pip install TTS trainer numpy==1.23.5 mutagen


# Code Starts
# Import Libraries
wav_path = "/kaggle/input/hindi-small/train/audio/"
# txt_path = "/kaggle/input/tts-hindi-f/Hindi-F/txt"
transcript_path = "/kaggle/working/transcription1.txt"
dataset_path = "/kaggle/input/tts-hindi-f/Hindi-F"
transcript_name = "transcription1.txt"
model_path = "/kaggle/working/Models"



# Use a set to store unique texts
# unique_texts = set()

# # Open the input file and read it line by line
# with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
#     for line in infile:
#         # Split the line into video ID and text parts
#         parts = line.split('@')
#         if len(parts) == 2:
#             video_id = parts[0]
#             text = parts[1].strip()
            
#             # Check if the text is already in the set of unique texts
#             if text not in unique_texts:
#                 # If the text is not in the set, add it and write the line to the output file
#                 unique_texts.add(text)
#                 outfile.write(line)



input_file_path = '/kaggle/input/hindi-small/train/transcription.txt'  # Path to your input file
output_file_path = '/kaggle/working/transcription.txt'  # Path to the output file where modified lines will be saved

all_text = ""

# Open the input file and read it line by line
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Replace the first space with an '@'
        modified_line = line.replace(' ', '@', 1)
        # Write the modified line to the output file
        outfile.write(modified_line)

print("File processing complete. Modified lines saved to output.txt")


unique_texts =  set()

with open(output_file_path, 'r', encoding='utf-8') as infile, open("/kaggle/working/transcription1.txt", 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Split the line into video ID and text parts
        parts = line.split('@')
        if len(parts) == 2:
            video_id = parts[0]
            text = parts[1].strip()
            all_text+=text
            
            # Check if the text is already in the set of unique texts
            if text not in unique_texts:
                # If the text is not in the set, add it and write the line to the output file
                unique_texts.add(text)
                outfile.write(line)
                
chars = ""
for i in set(all_text):
    chars+=i
    
# print(chars)

all_chars = ''.join(sorted(list(chars)))




import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


import os
from tqdm import tqdm

# !mkdir Models
output_path = model_path

dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train=transcript_name, path=dataset_path
)

audio_config = VitsAudioConfig(
    sample_rate=44100, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

import os
import regex


character_config = CharactersConfig(
    characters_class= "TTS.tts.models.vits.VitsCharacters",
    characters= all_chars,
    punctuations= " ?!,-|.",
    pad= "<PAD>",
    eos= "<EOS>",
    bos= "<BOS>",
    blank= "<BLNK>",
)

config = VitsConfig(
    audio=audio_config,
    characters=character_config,
    run_name="Hindi-TTS-01",
    batch_size=16,
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=0,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    save_best_after=1000,
    save_checkpoints=True,
    save_all_best=True,
    mixed_precision=True,
    max_text_len=250,  # change this if you have a larger VRAM than 16GB
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    test_sentences=[
        ["मेरा नाम क्या है"],
        ["हेलो बरोथेर और क्या चल रहा"],
        ["ईशा अल्लाह लड़कों ने अच्छा खेला"]
    ]
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

path_wav= wav_path


def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = transcript_path
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("@")
            wav_file = f"{path_wav}{cols[0]}.wav"
            text = cols[1]
            # print(text)
            items.append({"text":text, "audio_file":wav_file, "speaker_name":cols[0][5:8], "root_path": root_path})
    return items

train_samples, eval_samples = load_tts_samples(
dataset_config,
eval_split=True,
formatter=formatter)



# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
