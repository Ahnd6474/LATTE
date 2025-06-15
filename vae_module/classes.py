from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class Tokenizer:
    vocab: Sequence[str]
    pad_token: str = "<pad>"
    bos_token: str = "<cls>"

    def __post_init__(self):
        self.idx_to_tok = list(self.vocab)
        self.tok_to_idx = {t: i for i, t in enumerate(self.idx_to_tok)}
        self.pad_idx = self.tok_to_idx[self.pad_token]
        self.bos_idx = self.tok_to_idx[self.bos_token]

    def get_idx(self, token: str) -> int:
        return self.tok_to_idx[token]

    def get_tok(self, idx: int) -> str:
        return self.idx_to_tok[idx]


class SequenceDataset(Dataset):
    """Dataset wrapping a list of string sequences."""

    def __init__(self, sequences: List[str], tokenizer: Tokenizer, max_len: int):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from .utils import sequence_to_tensor

        seq = self.sequences[idx]
        return sequence_to_tensor(seq, self.tokenizer, self.max_len)
