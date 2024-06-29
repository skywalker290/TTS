# !pip install TTS trainer numpy==1.23.5 mutagen


# Code Starts
# Import Libraries
wav_path = "/kaggle/input/tts-hindi-f/Hindi-F/wav"
txt_path = "/kaggle/input/tts-hindi-f/Hindi-F/txt"
transcript_path = "/kaggle/input/tts-hindi-f/transcript (1).txt"
dataset_path = "/kaggle/input/tts-hindi-f/Hindi-F"
transcript_name = "transcript (1).txt"
model_path = "/kaggle/working/Models"


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

# Function to get all Devanagari characters
def get_devanagari_characters():
    devanagari_characters = set()
    for codepoint in range(0x0900, 0x097F + 1):
        character = chr(codepoint)
        if regex.match(r'\p{Devanagari}', character):
            devanagari_characters.add(character)
    return devanagari_characters

def get_unique_characters_and_punctuations(directory):
    devanagari_characters = get_devanagari_characters()
    unique_characters = set()
    unique_punctuations = set()

    # Define a set of common punctuation marks
    common_punctuations = set(" !,.?-।")

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                for char in text:
                    if char in common_punctuations:
                        unique_punctuations.add(char)
                    elif char in devanagari_characters:
                        unique_characters.add(char)

    return unique_characters, unique_punctuations


def create_characters_config(directory):
    unique_characters, unique_punctuations = get_unique_characters_and_punctuations(directory)

    characters = "".join(sorted(unique_characters))
    punctuations = "".join(sorted(unique_punctuations))

    return {'chars':characters,'punc':punctuations}

# Specify the directory containing the .txt files
directory = txt_path
chars_punc = create_characters_config(directory)

print("Characters Config:")
print(chars_punc)


character_config = CharactersConfig(
    characters_class= "TTS.tts.models.vits.VitsCharacters",
    characters= chars_punc['chars'],
    punctuations= chars_punc['punc'],
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
            wav_file = f"{path_wav}/{cols[0]}.wav"
            text = cols[1]
            # print(text)
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items

train_samples, eval_samples = load_tts_samples(
dataset_config,
eval_split=True,
formatter=formatter)


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
