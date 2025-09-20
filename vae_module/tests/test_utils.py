import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vae_module.classes import Tokenizer
from vae_module.exceptions import InvalidSequenceError, SequenceLengthError
from vae_module.utils import sequence_to_tensor


def make_tokenizer():
    return Tokenizer(["<cls>", "<pad>", "A", "B", "C"])


def test_sequence_to_tensor_raises_on_unknown_token():
    tokenizer = make_tokenizer()
    with pytest.raises(InvalidSequenceError):
        sequence_to_tensor("AD", tokenizer, max_len=2)


def test_sequence_to_tensor_strict_length_check():
    tokenizer = make_tokenizer()
    with pytest.raises(SequenceLengthError):
        sequence_to_tensor("ABC", tokenizer, max_len=2)


def test_sequence_to_tensor_truncates_when_not_strict():
    tokenizer = make_tokenizer()
    tensor = sequence_to_tensor("ABC", tokenizer, max_len=2, strict=False)
    assert torch.equal(
        tensor,
        torch.tensor([
            tokenizer.get_idx("A"),
            tokenizer.get_idx("B"),
        ]),
    )
