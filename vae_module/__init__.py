"""VAE wrapper module with logging and custom exceptions."""

from .config import Config, load_config
from .loader import load_vae
from .encoder import encode, encode_batch, encode_long, encode_long_batch
from .decoder import decode, decode_batch
from .classes import SequenceDataset, Tokenizer
from .utils import sequence_to_tensor, tensor_to_sequence, pad_collate
from .logger import setup_logger
from .exceptions import (
    VAEError,
    InvalidSequenceError,
    SequenceLengthError,
    DeviceNotAvailableError,
)

__all__ = [
    "Config",
    "load_config",
    "load_vae",
    "encode",
    "encode_batch",
    "encode_long",
    "encode_long_batch",
    "decode",
    "decode_batch",
    "SequenceDataset",
    "Tokenizer",
    "sequence_to_tensor",
    "tensor_to_sequence",
    "pad_collate",
    "setup_logger",
    "VAEError",
    "InvalidSequenceError",
    "SequenceLengthError",
    "DeviceNotAvailableError",
]
