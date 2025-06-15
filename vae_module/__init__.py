"""VAE wrapper module with logging and custom exceptions."""

from .config import Config, load_config
from .loader import load_vae
from .encoder import encode, encode_batch
from .decoder import decode, decode_batch
from .classes import SequenceDataset, Tokenizer
from .utils import sequence_to_tensor, tensor_to_sequence
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
    "decode",
    "decode_batch",
    "SequenceDataset",
    "Tokenizer",
    "sequence_to_tensor",
    "tensor_to_sequence",
    "setup_logger",
    "VAEError",
    "InvalidSequenceError",
    "SequenceLengthError",
    "DeviceNotAvailableError",
]
