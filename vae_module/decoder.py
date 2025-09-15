from typing import List, Sequence, Optional

import torch
from tqdm import trange

from .logger import setup_logger
from .model import VAEWithSurrogate
from .classes import Tokenizer
from .utils import tensor_to_sequence

logger = setup_logger(__name__)
def _band_memory_mask(t: int, mem_len: int, band: int, device):
    # (t, mem_len): 밴드 안=0.0, 바깥=-inf (가산 마스크)
    idx_t  = torch.arange(t, device=device).unsqueeze(1)      # (t,1)
    idx_m  = torch.arange(mem_len, device=device).unsqueeze(0) # (1,mem_len)
    dist   = (idx_t - idx_m).abs()
    mask   = torch.full((t, mem_len), 0.0, device=device)
    mask[dist > band] = float("-inf")
    return mask
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
    use_diag_band: bool = True,     # ★ 추가: 대각선 밴드 마스크 사용
    band_width: int = 8             # ★ 추가: 허용 밴드 폭 |t-i|≤w
) -> str:
    model.eval()
    device = next(model.parameters()).device
    _assert_tokenizer_and_embed(tokenizer, model)

    z = z.flatten().unsqueeze(0).to(device)

    generated = torch.full((1, max_len), tokenizer.pad_idx, device=device, dtype=torch.long)
    generated[:, 0] = tokenizer.bos_idx

    full_tgt_mask = torch.triu(
        torch.full((max_len, max_len), float("-inf"), device=device), diagonal=1
    )

    memory, mem_pad_mask = build_memory_from_z(model, z, max_len, x_gt_ids, use_surrogate=surrogate)
    mem_len = memory.size(1)

    for t in range(1, max_len):
        tok_emb = model.dec_emb(generated[:, :t])
        pos_emb = model.dec_pos[:, :t, :]
        z_emb  = model.latent2emb(z).unsqueeze(1).expand(-1, t, -1)
        tgt = tok_emb + pos_emb + z_emb

        # ★★ 핵심: 대각선 밴드 메모리 마스크 (t, mem_len)
        mem_bias = None
        if use_diag_band:
            mem_bias = _band_memory_mask(t, mem_len, band_width, device)

        dec_out = model.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=full_tgt_mask[:t, :t],
            memory_mask=mem_bias,               # ★ 여기!
            memory_key_padding_mask=mem_pad_mask,
        )
        dec_out = _ensure_batch_time(dec_out, B_expected=generated.size(0))
        step_logits = model.out(dec_out)[:, -1, :]

        if top_k is None and (temperature is None or temperature == 1.0):
            next_token = step_logits.argmax(-1)
        else:
            temp = 1.0 if (temperature is None) else float(temperature)
            if temp <= 0:
                raise ValueError("temperature must be > 0")
            step_logits = step_logits / temp
            if top_k is not None:
                k = max(1, min(int(top_k), step_logits.size(-1)))
                vals, idxs = torch.topk(step_logits, k, dim=-1)
                masked = torch.full_like(step_logits, float("-inf"))
                masked.scatter_(1, idxs, vals)
                step_logits = masked
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
