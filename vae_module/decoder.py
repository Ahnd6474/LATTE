from typing import List, Optional, Sequence, Tuple

import torch

from .classes import Tokenizer
from .logger import setup_logger
from .model import VAEWithSurrogate
from .utils import tensor_to_sequence

logger = setup_logger(__name__)


def _assert_tokenizer_and_embed(tokenizer: Tokenizer, model: VAEWithSurrogate) -> None:
    assert tokenizer.pad_idx != tokenizer.bos_idx != getattr(
        tokenizer, "eos_idx", -9999
    ), (
        "PAD/BOS/EOS index overlap? "
        f"pad={tokenizer.pad_idx}, bos={tokenizer.bos_idx}, eos={getattr(tokenizer, 'eos_idx', None)}"
    )
    if hasattr(model, "dec_emb") and hasattr(model.dec_emb, "padding_idx"):
        assert model.dec_emb.padding_idx == tokenizer.pad_idx, (
            f"dec_emb.padding_idx({model.dec_emb.padding_idx}) != "
            f"tokenizer.pad_idx({tokenizer.pad_idx})"
        )


def _prepare_latent(z: torch.Tensor, device: torch.device) -> torch.Tensor:
    z = z.to(device)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    return z


def build_memory_from_z(
    model: VAEWithSurrogate,
    z: torch.Tensor,
    max_len: int,
    x_gt_ids: Optional[torch.Tensor],
    use_surrogate: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct decoder memory given a latent vector."""

    if use_surrogate and getattr(model, "build_surrogate_memory", None) is not None:
        return model.build_surrogate_memory(z, x_gt_ids=x_gt_ids)

    if x_gt_ids is None:
        raise ValueError("x_gt_ids must be provided when surrogate decoding is disabled")

    if x_gt_ids.dim() == 1:
        x_gt_ids = x_gt_ids.unsqueeze(0)
    x_gt_ids = x_gt_ids[:, :max_len].to(z.device)
    memory, enc_mask = model.encoder(x_gt_ids)
    mem_pad_mask = ~enc_mask
    return memory, mem_pad_mask


@torch.no_grad()
def decode(
    model,
    z: torch.Tensor,
    tokenizer,
    max_len: int,
    truncate_len: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    surrogate: bool = True,
    x_gt_ids: Optional[torch.Tensor] = None,
    use_diag_band: bool = True,
    band_width: int = 8,
) -> str:
    """Decode a single latent vector into a sequence."""

    del band_width  # kept for backward compatibility

    model.eval()
    device = next(model.parameters()).device
    _assert_tokenizer_and_embed(tokenizer, model)

    z = _prepare_latent(z, device)
    model._z_cached = z

    memory, mem_pad_mask = build_memory_from_z(
        model, z, max_len, x_gt_ids=x_gt_ids, use_surrogate=surrogate
    )

    generated = torch.full((z.size(0), max_len), tokenizer.pad_idx, device=device, dtype=torch.long)
    generated[:, 0] = tokenizer.bos_idx

    use_bias = bool(model.diag_bias) if use_diag_band else False

    for t in range(1, max_len):
        logits = model.decode_step(
            generated[:, :t],
            memory,
            mem_pad_mask,
            tokenizer=tokenizer,
            use_bias=use_bias,
        )

        if temperature is None or temperature == 1.0:
            step_logits = logits
        else:
            temp = float(temperature)
            if temp <= 0:
                raise ValueError("temperature must be > 0")
            step_logits = logits / temp

        if top_k is not None:
            k = max(1, min(int(top_k), step_logits.size(-1)))
            vals, idxs = torch.topk(step_logits, k, dim=-1)
            masked = torch.full_like(step_logits, float("-inf"))
            masked.scatter_(1, idxs, vals)
            step_logits = masked

        if top_k is None and (temperature is None or temperature == 1.0):
            next_token = step_logits.argmax(-1)
        else:
            probs = torch.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

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
    """Decode a batch of latent vectors."""

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
