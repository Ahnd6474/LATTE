from typing import List

import torch

from .exceptions import InvalidSequenceError, SequenceLengthError


def sequence_to_tensor(seq: str, tokenizer: "Tokenizer", max_len: int) -> torch.LongTensor:
    """Convert a string sequence to tensor of token IDs."""
    if any(c not in tokenizer.vocab for c in seq):
        raise InvalidSequenceError(seq)
    if len(seq) > max_len:
        raise SequenceLengthError(len(seq), max_len)
    ids = [tokenizer.get_idx(c) for c in seq]
    return torch.tensor(ids, dtype=torch.long)


def tensor_to_sequence(tensor: torch.Tensor, tokenizer: "Tokenizer") -> str:
    """Convert tensor of token IDs back to string sequence."""
    chars = [tokenizer.get_tok(int(i)) for i in tensor.tolist() if i != tokenizer.pad_idx]
    return "".join(chars)
