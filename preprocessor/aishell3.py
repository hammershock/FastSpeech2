import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from typings_ import PreprocessConfig


def prepare_align(config: PreprocessConfig) -> None:
    """
    Corpus --> raw
    Input files:
        in_dir(corpus_path): config["path"]["corpus_path"]:
            corpus_path/[train/test]/content.txt
            corpus_path/[train/test]/wav/{speaker}/{wave_filename}.wav
    Output files:
        out_dir(raw_path): config["path"]["raw_path"]:
            raw_path/speaker/{wave_filename}.lab
            raw_path/speaker/{wave_filename}.wav

    Preprocess Data and Generate .lab pinyin splits files

    :param config:
    in-files: content.txt

    """
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]

    # sr=22050, max_wav_value=32768
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

    for dataset in ["train", "test"]:
        print(f"Processing {dataset}ing set...")
        # "./AISHELL-3/train/content.txt"
        """
        The format of AISHELL-Format content is like:
        SSB00050001.wav	广 guang3 州 zhou1 女 nv3 大 da4 学 xue2 生 sheng1 登 deng1 山 shan1 失 shi1 联 lian2 四 si4 天 tian1 警 jing3 方 fang1 找 zhao3 到 dao4 疑 yi2 似 si4 女 nv3 尸 shi1
        SSB00050002.wav	尊 zhun1 重 zhong4 科 ke1 学 xue2 规 gui1 律 lv4 的 de5 要 yao1 求 qiu2
        """
        content_path = os.path.join(in_dir, dataset, "content.txt")
        with open(content_path, encoding="utf-8") as f:
            # iteration over content line
            for line in tqdm(f):
                wav_name, text = line.strip("\n").split("\t")
                speaker = wav_name.split("/")[0]  # speaker name
                text = text.split(" ")[1::2]  # text pinyins

                wav_path = os.path.join(in_dir, dataset, "wav", wav_name)
                if os.path.exists(wav_path):
                    # load wav file
                    wav, _ = librosa.load(wav_path, sr=sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value  # scaling
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    # write back audio
                    basename = os.path.splitext(wav_name)[0]
                    wav_out_path = os.path.join(out_dir, wav_name)
                    wavfile.write(wav_out_path, sampling_rate, wav.astype(np.int16))
                    lab_out_path = os.path.join(out_dir, f"{basename}.lab")
                    with open(lab_out_path, "w") as f1:
                        # split the pinyins with space
                        f1.write(" ".join(text))
