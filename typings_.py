from typing import TypedDict, List


__all__ = ['TrainConfig', 'PreprocessConfig', "ModelConfig"]


class PathConfig(TypedDict):
    ckpt_path: str
    log_path: str
    result_path: str


class OptimizerConfig(TypedDict):
    batch_size: int
    betas: List[float]
    eps: float
    weight_decay: float
    grad_clip_thresh: float
    grad_acc_step: int
    warm_up_step: int
    anneal_steps: List[int]
    anneal_rate: float


class StepConfig(TypedDict):
    total_step: int
    log_step: int
    synth_step: int
    val_step: int
    save_step: int


class TrainConfig(TypedDict):
    path: PathConfig
    optimizer: OptimizerConfig
    step: StepConfig


from typing import TypedDict, List, Optional


class PathConfig(TypedDict):
    corpus_path: str
    lexicon_path: str
    raw_path: str
    preprocessed_path: str


class TextConfig(TypedDict):
    text_cleaners: List[str]
    language: str


class AudioConfig(TypedDict):
    sampling_rate: int
    max_wav_value: float


class STFTConfig(TypedDict):
    filter_length: int
    hop_length: int
    win_length: int


class MelConfig(TypedDict):
    n_mel_channels: int
    mel_fmin: int
    mel_fmax: Optional[int]  # Use Optional to allow None


class PitchConfig(TypedDict):
    feature: str
    normalization: bool


class EnergyConfig(TypedDict):
    feature: str
    normalization: bool


class PreprocessingConfig(TypedDict):
    val_size: int
    text: TextConfig
    audio: AudioConfig
    stft: STFTConfig
    mel: MelConfig
    pitch: PitchConfig
    energy: EnergyConfig


class PreprocessConfig(TypedDict):
    dataset: str
    path: PathConfig
    preprocessing: PreprocessingConfig


from typing import TypedDict, List


class TransformerConfig(TypedDict):
    encoder_layer: int
    encoder_head: int
    encoder_hidden: int
    decoder_layer: int
    decoder_head: int
    decoder_hidden: int
    conv_filter_size: int
    conv_kernel_size: List[int]
    encoder_dropout: float
    decoder_dropout: float


class VariancePredictorConfig(TypedDict):
    filter_size: int
    kernel_size: int
    dropout: float


class VarianceEmbeddingConfig(TypedDict):
    pitch_quantization: str
    energy_quantization: str
    n_bins: int


# class GSTConfig(TypedDict, total=False):
#     use_gst: bool
#     conv_filters: List[int]
#     gru_hidden: int
#     token_size: int
#     n_style_token: int
#     attn_head: int

class VocoderConfig(TypedDict):
    model: str
    speaker: str


class ModelConfig(TypedDict):
    transformer: TransformerConfig
    variance_predictor: VariancePredictorConfig
    variance_embedding: VarianceEmbeddingConfig
    # gst: GSTConfig  # Uncomment and use if GST configuration is needed
    multi_speaker: bool
    max_seq_len: int
    vocoder: VocoderConfig
