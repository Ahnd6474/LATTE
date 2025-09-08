from typing import List, Sequence, Optional

import torch
from tqdm import trange

from .logger import setup_logger
from .model import VAEWithSurrogate
from .classes import Tokenizer
from .utils import tensor_to_sequence

logger = setup_logger(__name__)


def decode(
    model: VAEWithSurrogate,
    z: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int,
    truncate_len: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
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
    temperature:
        Sampling temperature. ``1.0`` uses unscaled logits; values other than
        ``1.0`` enable stochastic sampling.
    top_k:
        If provided, restrict sampling to the top ``k`` tokens at each step.
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

    # Prepare decoder memory using the surrogate if available
    if getattr(model, "surrogate", None) is not None:
        mask_bool = torch.ones(1, max_len, dtype=torch.bool, device=device)
        memory, mem_mask = model.surrogate(z, mask_bool, causal_self=False)
    else:
        memory = torch.zeros(1, max_len, model.dec_emb.embedding_dim, device=device)
        mem_mask = torch.ones(1, max_len, dtype=torch.bool, device=device)

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
                memory_key_padding_mask=~mem_mask,
            )

            logits = model.out(dec_out)
            step_logits = logits[:, -1, :]

            if top_k is None and temperature == 1.0:
                next_token = step_logits.argmax(-1)
            else:
                step_logits = step_logits / temperature
                if top_k is not None:
                    k = min(top_k, step_logits.size(-1))
                    values, indices = torch.topk(step_logits, k)
                    mask = torch.full_like(step_logits, float("-inf"))
                    mask.scatter_(1, indices, values)
                    step_logits = mask
                probs = torch.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)

            generated[:, t] = next_token
            if (next_token == tokenizer.pad_idx).all():
                break

    ids = generated[0]
    seq = tensor_to_sequence(ids, tokenizer)
    if truncate_len is not None:
        seq = seq[:truncate_len]
    return seq


def decode_batch(
    model: VAEWithSurrogate,
    Z: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int,
    truncate_lens: Optional[Sequence[int]] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> List[str]:
    """Decode a batch of latent vectors.

    Parameters
    ----------
    truncate_lens:
        Optional sequence of lengths used to truncate each decoded sequence.
        If ``None`` (default), sequences are returned unmodified.
    temperature:
        Sampling temperature passed to :func:`decode`.
    top_k:
        Top-k value passed to :func:`decode`.
    """

    if truncate_lens is None:
        return [
            decode(
                model,
                z,
                tokenizer,
                max_len,
                temperature=temperature,
                top_k=top_k,
            )
            for z in Z
        ]

    if len(truncate_lens) != len(Z):
        raise ValueError("truncate_lens must match batch size")

    return [
        decode(
            model,
            z,
            tokenizer,
            max_len,
            tlen,
            temperature=temperature,
            top_k=top_k,
        )
        for z, tlen in zip(Z, truncate_lens)
    ]
