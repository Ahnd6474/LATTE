from dataclasses import dataclass
from typing import ClassVar, List, Sequence
from typing import List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class Tokenizer:
    """Simple vocabulary wrapper with ESM token defaults."""

    vocab: Sequence[str]
    pad_token: str = "<pad>"
    bos_token: str = "<cls>"
    mask_token: str = "<mask>"

    # ESM alphabet copied from ``fair-esm`` constants for ``ESM-1b`` / ``ESM2``.
    ESM_STANDARD_TOKS: ClassVar[Sequence[str]] = [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
    ]

    @classmethod
    def from_esm(cls) -> "Tokenizer":
        """Return a tokenizer initialized with the ESM-1b alphabet."""

        toks = ["<cls>", "<pad>", "<eos>", "<unk>"] + list(cls.ESM_STANDARD_TOKS)
        if len(toks) % 8:
            toks.append("<null_1>")
        toks.append("<mask>")
        return cls(toks)
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
