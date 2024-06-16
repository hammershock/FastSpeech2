"""
Prepare Genshin Dataset

Just like the AISHELL3 Dataset!
"""

import os
import shutil
from collections import defaultdict
import random

from tqdm import tqdm
import pypinyin

speaker_names = os.listdir("GenshinVoiceRaw")


class Pair:
    def __init__(self):
        self.wav_path = None
        self.content = None
        self.anno_line = None
        self.speaker_name = None

    def __repr__(self):
        return f"{self.anno_line}"


# get all the filename.wav, filename.lab pairs
filepaths = defaultdict(Pair)

for speaker_name in speaker_names:
    speaker_dir = os.path.join("GenshinVoiceRaw", speaker_name)

    for root, dirs, files in os.walk(speaker_dir):
        for file in tqdm(files, f"processing {speaker_name}"):
            basename = os.path.splitext(file)[0]
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                filepaths[basename].wav_path = wav_path

            if file.endswith(".lab"):
                lab_path = os.path.join(root, file)
                with open(lab_path, "r") as f:
                    content = "".join(line for line in f)
                filepaths[basename].content = content
                filepaths[basename].speaker_name = speaker_name


# now we got several .wav/.lab pairs
# differ from AISHELL3 Dataset, there is no content.txt to store the metadata...
# so we need to construct content.txt from .lab files
output_dir = "./GenshinVoice"
# os.makedirs(output_dir, exist_ok=True)
pairs = list(filepaths.values())
print(len(pairs))
for pair in pairs:
    filename = os.path.basename(pair.wav_path)
    text = pair.content
    chinese_text = ''.join(filter(lambda ch: '\u4e00' <= ch <= '\u9fff', text))
    pinyin_pairs = pypinyin.pinyin(chinese_text, style=pypinyin.Style.TONE3, strict=False, neutral_tone_with_five=True)
    pinyin_text = ' '.join(f"{ch} {py[0]}" for ch, py in zip(chinese_text, pinyin_pairs))
    pair.anno_line = f"{pair.speaker_name}/{filename}\t{pinyin_text}"
random.shuffle(pairs)

train_split = int(len(pairs) * 0.8)
train_pairs = pairs[:train_split]
test_pairs = pairs[train_split:]

for dataset_type, pairs in [("train", train_pairs), ("test", test_pairs)]:
    os.makedirs(os.path.join(output_dir, dataset_type), exist_ok=True)
    for pair in tqdm(pairs):
        filename = os.path.basename(pair.wav_path)
        output_wav_dir = os.path.join(output_dir, dataset_type, "wav", pair.speaker_name)
        os.makedirs(output_wav_dir, exist_ok=True)
        shutil.copy(pair.wav_path, os.path.join(output_wav_dir, filename))
        with open(os.path.join(output_dir, dataset_type, "content.txt"), "a") as f:
            f.write(pair.anno_line + "\n")
            f.flush()
