from typing import List

import torch
from torch.utils.data import DataLoader

from .exceptions import InvalidSequenceError, SequenceLengthError
from .logger import setup_logger
from .utils import sequence_to_tensor
from .classes import Tokenizer
from .model import VAETransformerDecoder

logger = setup_logger(__name__)


def encode(model: VAETransformerDecoder, seq: str, tokenizer: Tokenizer, max_len: int) -> torch.Tensor:
    """Encode a single sequence into a latent vector."""
    model.eval()
    with torch.no_grad():
        x = sequence_to_tensor(seq, tokenizer, max_len).unsqueeze(0).to(next(model.parameters()).device)
        mask = x != tokenizer.pad_idx
        _, mu, logvar, *_ = model(x, mask)
        z = mu #+ torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logger.debug("Encoded sequence length %d", len(seq))
        return z.squeeze(0)


def encode_batch(model: VAETransformerDecoder, loader: DataLoader, tokenizer: Tokenizer) -> torch.Tensor:
    """Encode a batch of sequences from a dataloader."""
    model.eval()
    zs = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x in loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            mask = x != tokenizer.pad_idx
            _, mu, logvar, *_ = model(x, mask)
            z = mu #+ torch.randn_like(mu) * torch.exp(0.5 * logvar)
            zs.append(z.cpu())
    return torch.cat(zs, dim=0)


def encode_long(
    model: VAETransformerDecoder,
    seq: str,
    tokenizer: Tokenizer,
    max_len: int,
    overlap: int = 256,
) -> torch.Tensor:
    """Encode a long sequence by sliding a window with overlap.

    Each window has length ``max_len`` and successive windows start
    ``max_len - overlap`` characters after the previous one. The final
    window is padded with the pad token if needed. The latent vectors for
    all windows are stacked in order of appearance.
    """
    if overlap >= max_len:
        raise ValueError("overlap must be smaller than max_len")

    if len(seq) <= max_len:
        return encode(model, seq, tokenizer, max_len).unsqueeze(0)

    step = max_len - overlap
    segments = []
    for start in range(0, len(seq), step):
        end = start + max_len
        part = seq[start:end]
        t = sequence_to_tensor(part, tokenizer, max_len)
        if t.size(0) < max_len:
            pad_len = max_len - t.size(0)
            t = torch.nn.functional.pad(t, (0, pad_len), value=tokenizer.pad_idx)
            segments.append(t)
            break
        segments.append(t)
        if end >= len(seq):
            break

    device = next(model.parameters()).device
    x = torch.stack(segments).to(device)
    mask = x != tokenizer.pad_idx
    model.eval()
    with torch.no_grad():
        _, mu, logvar, *_ = model(x, mask)
        z = mu
    return z.cpu()


def encode_long_batch(
    model: VAETransformerDecoder,
    sequences: List[str],
    tokenizer: Tokenizer,
    max_len: int,
    overlap: int = 256,
) -> List[torch.Tensor]:
    """Encode a batch of long sequences using :func:`encode_long`."""
    return [encode_long(model, s, tokenizer, max_len, overlap) for s in sequences]

