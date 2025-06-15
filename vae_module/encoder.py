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
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
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
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            zs.append(z.cpu())
    return torch.cat(zs, dim=0)
