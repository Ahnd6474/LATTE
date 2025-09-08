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
    surrogate: bool = True,                 # ★ 추가: 메모리를 surrogate로 쓸지 토글
    x_gt_ids: Optional[torch.Tensor] = None # (옵션) GT 길이 마스크를 쓰고 싶으면 제공
) -> str:
    """Free-run AR decoding. surrogate=True면 surrogate(z) 메모리를 사용."""
    model.eval()
    device = next(model.parameters()).device
    z = z.unsqueeze(0).to(device)

    # 생성 버퍼
    generated = torch.full(
        (1, max_len), tokenizer.pad_idx, device=device, dtype=torch.long
    )
    generated[:, 0] = tokenizer.bos_idx

    # causal mask
    full_tgt_mask = torch.triu(
        torch.full((max_len, max_len), float("-inf"), device=device), diagonal=1
    )

    # ── memory 구성 ─────────────────────────────────────────────
    if surrogate and getattr(model, "surrogate", None) is not None:
        # (옵션) x_gt_ids가 있으면 encoder에서 pad 마스크만 가져와 surrogate에 전달
        if x_gt_ids is not None:
            if x_gt_ids.dim() == 1:
                x_gt_ids = x_gt_ids.unsqueeze(0)
            x_gt_ids = x_gt_ids.to(device)
            with torch.no_grad():
                _, enc_mask = model.encoder(x_gt_ids)   # GT pad 마스크
            mem_mask = enc_mask
        else:
            mem_mask = torch.ones(1, max_len, dtype=torch.bool, device=device)

        memory, _ = model.surrogate(z, mem_mask, causal_self=False)  # ★ surrogate 경로
        # 어댑터 없음: 그대로 메모리 사용
    else:
        # surrogate 안 쓰면 zero-memory
        memory = torch.zeros(1, max_len, model.dec_emb.embedding_dim, device=device)
        mem_mask = torch.ones(1, max_len, dtype=torch.bool, device=device)
    # ───────────────────────────────────────────────────────────

    # AR 생성 루프
    with torch.no_grad():
        for t in range(1, max_len):
            tok_emb = model.dec_emb(generated[:, :t])
            pos_emb = model.dec_pos[:, :t, :]
            z_emb = model.latent2emb(z).unsqueeze(1).expand(-1, t, -1)
            tgt = tok_emb + pos_emb + z_emb

            dec_out = model.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=full_tgt_mask[:t, :t],
                memory_key_padding_mask=~mem_mask,  # 패딩만 차단(비-causal)
            )

            step_logits = model.out(dec_out)[:, -1, :]

            # 샘플링
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
            # (원 코드 유지) pad 나오면 종료
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
