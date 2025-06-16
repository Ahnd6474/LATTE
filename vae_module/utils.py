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


def pad_collate(batch: List[torch.Tensor], pad_idx: int) -> torch.Tensor:
    """Pad a list of variable length tensors for DataLoader batching."""
    if not batch:
        return torch.empty(0, dtype=torch.long)
    max_len = max(t.size(0) for t in batch)
    padded = [
        torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=pad_idx)
        for t in batch
    ]
    return torch.stack(padded)
