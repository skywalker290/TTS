# !pip install TTS numpy==1.23.5
# !python -m trainer.distribute --script /kaggle/working/VITS/new.py --gpus "0,1"

import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"
from dataclasses import dataclass, field

from trainer import Trainer, TrainerArgs

from TTS.config import load_config, register_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model



@dataclass
class TrainTTSArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={"help": "Path to the config file."})

# def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
#     """Assumes each line as ```<filename>|<transcription>```
#     """
#     txt_file = "/kaggle/input/tts-hindi-f/transcript (1).txt"
#     items = []
#     speaker_name = "my_speaker"
#     with open(txt_file, "r", encoding="utf-8") as ttf:
#         for line in ttf:
#             cols = line.split("@")
#             path = "/kaggle/input/tts-hindi-f/Hindi-F/wav/"
#             wav_file = f"{path}{cols[0]}.wav"
#             text = cols[1]
#             # print(text)
#             items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
#     return items
wav_path = "/kaggle/input/hindi-small/train/audio/"
# txt_path = "/kaggle/input/tts-hindi-f/Hindi-F/txt"
transcript_path = "/kaggle/working/transcription1.txt"
dataset_path = "/kaggle/input/tts-hindi-f/Hindi-F"
transcript_name = "transcription1.txt"
model_path = "/kaggle/working/Models"

def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = transcript_path
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("@")
            wav_file = f"{wav_path}{cols[0]}.wav"
            text = cols[1]
            # print(text)
            items.append({"text":text, "audio_file":wav_file, "speaker_name":cols[0][5:8], "root_path": root_path})
    return items

def main():
    """Run `tts` model training directly by a `config.json` file."""
    # init trainer args
    train_args = TrainTTSArgs()
    parser = train_args.init_argparse(arg_prefix="")

    # override trainer args from comman-line args
    args, config_overrides = parser.parse_known_args()
    train_args.parse_args(args)
    

    # load config.json and register
    # args.continue_path = "/kaggle/working/tts/Hindi-TTS-01-June-27-2024_03+00PM-0000000"
    
    # if args.config_path or args.continue_path:
    #     if args.config_path:
    #         # init from a file
    #         config = load_config(args.config_path)
    #         if len(config_overrides) > 0:
    #             config.parse_known_args(config_overrides, relaxed_parser=True)
    #     elif args.continue_path:
    #         # continue from a prev experiment
    #         config = load_config(os.path.join(args.continue_path, "config.json"))
    #         if len(config_overrides) > 0:
    #             config.parse_known_args(config_overrides, relaxed_parser=True)
    #     else:
    #         # init from console args
    #         from TTS.config.shared_configs import BaseTrainingConfig  # pylint: disable=import-outside-toplevel

    #         config_base = BaseTrainingConfig()
    #         config_base.parse_known_args(config_overrides)
    #         config = register_config(config_base.model)()

    
    
    # args.continue_path = "/kaggle/working/tts/Hindi-TTS-01-June-27-2024_03+00PM-0000000"
    
    config = load_config(os.path.join(args.continue_path, "config.json"))
    if len(config_overrides) > 0:
        config.parse_known_args(config_overrides, relaxed_parser=True)


    # load training samples
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=formatter
    )

    # init the model from config
    model = setup_model(config, train_samples + eval_samples)

    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        model.config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    trainer.fit()


if __name__ == "__main__":
    main()

