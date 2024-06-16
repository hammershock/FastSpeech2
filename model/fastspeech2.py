import json
import os

import torch.nn as nn

from transformer import Encoder, Decoder, PostNet
from utils.tools import get_mask_from_lengths
from .modules import VarianceAdaptor


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)

        decoder_hidden = model_config["transformer"]["decoder_hidden"]  # 256
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]  # 80

        self.mel_linear = nn.Linear(decoder_hidden, n_mel_channels)
        self.postnet = PostNet()

        self.speaker_emb = None

        if model_config["multi_speaker"]:
            # map of speaker ids
            speaker_config_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
            with open(speaker_config_path, "r") as f:
                n_speaker = len(json.load(f))
                encoder_hidden = model_config["transformer"]["encoder_hidden"]
            self.speaker_emb = nn.Embedding(n_speaker, encoder_hidden)
            # if multi-speakers available, initialize the speaker embeddings

    def forward(self, speakers, texts, src_lens, max_src_len,
                mels=None, mel_lens=None, max_mel_len=None,
                p_targets=None, e_targets=None, d_targets=None,
                p_control=1.0, e_control=1.0, d_control=1.0,
    ):
        # generate src-masks and tgt-masks
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None

        # text encoder
        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            # add some speaker embeddings
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        # variance adaptor
        output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks = \
            self.variance_adaptor(output, src_masks, mel_masks, max_mel_len, p_targets, e_targets, d_targets, p_control, e_control, d_control)

        # tgt mel-sequence output
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        # TODO: why must postnet?
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
