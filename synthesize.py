import argparse
import re
from string import punctuation
from typing import List, Dict

import numpy as np
import torch
import yaml
from pypinyin import pinyin, Style
from torch.utils.data import DataLoader

from dataset import TextDataset
from io_ import load_config
from text import text_to_sequence
from model.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from typings_ import TrainConfig, PreprocessConfig, ModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    from g2p_en import G2p

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(phones, preprocess_config["preprocessing"]["text"]["text_cleaners"])
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    # load vocabulary
    lexicon_path = preprocess_config["path"]["lexicon_path"]
    # split pinyin to 声母 + 韵母
    lexicon: Dict[str, List[str]] = read_lexicon(lexicon_path)

    pinyins = [p[0] for p in pinyin(text, style=Style.TONE3, strict=False, neutral_tone_with_five=True)]
    phones: list = []
    for p in pinyins:
        phones.extend(lexicon.get(p, ["sp"]))
    phones: str = "{" + " ".join(phones) + "}"
    print(f"Raw Text Sequence: {text}")
    print(f"Phoneme Sequence: {phones}")

    text_cleaners: list = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    sequence = np.array(text_to_sequence(phones, text_cleaners))
    return np.array(sequence)  # np.array [phoneme_len, ]


@torch.no_grad()
def synthesize(model, vocoder, batchs, *,
               preprocess_config: PreprocessConfig,
               model_config: ModelConfig,
               train_config: TrainConfig,
               pitch_control=1.0,
               energy_control=1.0,
               duration_control=1.0
               ):
    for batch in batchs:
        batch = to_device(batch, device)
        # Forward
        output = model(*(batch[2:]),
                       p_control=pitch_control,
                       e_control=energy_control,
                       d_control=duration_control
                       )

        # turn the model output into synthesize result
        synth_samples(batch, output, vocoder, model_config, preprocess_config, train_config["path"]["result_path"])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["batch", "single"], required=True,
                        help="Synthesize a whole dataset or a single sentence")
    parser.add_argument("--source", type=str, default=None,
                        help="path to a source file with format like train.txt and val.txt, for batch mode only")
    parser.add_argument("--text", type=str, default=None, help="raw text to synthesize, for single-sentence mode only")
    parser.add_argument("--speaker_id", type=int, default=0,
                        help="speaker ID for multi-speaker synthesis, for single-sentence mode only")
    parser.add_argument("-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str, required=True, help="path to model.yaml")
    parser.add_argument("-t", "--train_config", type=str, required=True, help="path to train.yaml")
    parser.add_argument("--pitch_control", type=float, default=1.0, help="pitch control")
    parser.add_argument("--energy_control", type=float, default=1.0, help="energy control")
    parser.add_argument("--duration_control", type=float, default=1.0, help="duration control")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Load Configs
    preprocess_config: PreprocessConfig = load_config(args.preprocess_config)
    model_config: ModelConfig = load_config(args.model_config)
    train_config: TrainConfig = load_config(args.train_config)

    # Get model
    model = get_model(args, device, train=False,
                      preprocess_config=preprocess_config,
                      model_config=model_config,
                      train_config=train_config)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # synthesize batch of sentences
    if args.mode == "batch":
        dataset = TextDataset(args.source, preprocess_config)  # Get dataset
        batchs = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)

    # synthesize single sentence
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        # speaker for each sentence
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        # texts: np.array [1, phoneme_len]
        # convert to phoneme ids
        text_lens = np.array([len(texts[0])])
        # text_lens: np.array [int, ...]
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    synthesize(model, vocoder, batchs,
               preprocess_config=preprocess_config,
               model_config=model_config,
               train_config=train_config,
               pitch_control=args.pitch_control,
               energy_control=args.energy_control,
               duration_control=args.duration_control)
