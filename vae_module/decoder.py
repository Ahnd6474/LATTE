from typing import List, Sequence, Optional

import torch
from tqdm import trange

from .logger import setup_logger
from .model import VAETransformerDecoder
from .classes import Tokenizer
from .utils import tensor_to_sequence

logger = setup_logger(__name__)


def decode(
    model: VAETransformerDecoder,
    z: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int,
    truncate_len: Optional[int] = None,
) -> str:
    """Decode a latent vector into a sequence.

    Parameters
    ----------
    model:
        Trained VAE model.
    z:
        Latent representation to decode.
    tokenizer:
        Tokenizer instance for mapping IDs to characters.
    max_len:
        Maximum length to generate.
    truncate_len:
        If provided, the decoded sequence will be truncated to this length.
    """
    model.eval()
    device = next(model.parameters()).device
    z = z.unsqueeze(0).to(device)

    generated = torch.full(
        (1, max_len), tokenizer.pad_idx, device=device, dtype=torch.long
    )
    generated[:, 0] = tokenizer.bos_idx

    tgt_mask = torch.triu(
        torch.full((max_len, max_len), float("-inf"), device=device), diagonal=1
    )
    memory = torch.zeros(1, max_len, model.dec_emb.embedding_dim, device=device)

    with torch.no_grad():
        for t in trange(1, max_len, disable=True):
            tok_emb = model.dec_emb(generated[:, :t])
            pos_emb = model.dec_pos[:, :t, :]
            z_emb = model.latent2emb(z).unsqueeze(1).expand(-1, t, -1)
            tgt = tok_emb + pos_emb + z_emb

            dec_out = model.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask[:t, :t],
            )

            logits = model.out(dec_out)
            next_token = logits[:, -1].argmax(-1)
            generated[:, t] = next_token
            if (next_token == tokenizer.pad_idx).all():
                break

    ids = generated[0]
    seq = tensor_to_sequence(ids, tokenizer)
    if truncate_len is not None:
        seq = seq[:truncate_len]
    return seq


def decode_batch(
    model: VAETransformerDecoder,
    Z: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int,
    truncate_lens: Optional[Sequence[int]] = None,
) -> List[str]:
    """Decode a batch of latent vectors.

    Parameters
    ----------
    truncate_lens:
        Optional sequence of lengths used to truncate each decoded sequence.
        If ``None`` (default), sequences are returned unmodified.
    """

    if truncate_lens is None:
        return [decode(model, z, tokenizer, max_len) for z in Z]

    if len(truncate_lens) != len(Z):
        raise ValueError("truncate_lens must match batch size")

    return [
        decode(model, z, tokenizer, max_len, tlen) for z, tlen in zip(Z, truncate_lens)
    ]
