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
    surrogate: bool = True,
    x_gt_ids: Optional[torch.Tensor] = None
) -> str:
    """Free-run AR decoding. surrogate=True면 surrogate(z) 메모리를 사용. (EOS 미사용)"""
    model.eval()
    device = next(model.parameters()).device
    z = z.unsqueeze(0).to(device)

    # 임베딩 padding_idx 안전 가드
    if hasattr(model, "dec_emb") and hasattr(model.dec_emb, "padding_idx"):
        assert model.dec_emb.padding_idx == tokenizer.pad_idx, \
            f"dec_emb.padding_idx({model.dec_emb.padding_idx}) != tokenizer.pad_idx({tokenizer.pad_idx})"

    # 생성 버퍼
    generated = torch.full((1, max_len), tokenizer.pad_idx, device=device, dtype=torch.long)
    generated[:, 0] = tokenizer.bos_idx

    # causal mask
    full_tgt_mask = torch.triu(
        torch.full((max_len, max_len), float("-inf"), device=device), diagonal=1
    )

    # ── memory 구성 ─────────────────────────────────────────────
    if surrogate and getattr(model, "surrogate", None) is not None:
        if x_gt_ids is not None:
            if x_gt_ids.dim() == 1:
                x_gt_ids = x_gt_ids.unsqueeze(0)
            x_gt_ids = x_gt_ids.to(device)
            with torch.no_grad():
                _, enc_mask = model.encoder(x_gt_ids)       # enc_mask: True==valid 가정
            mem_valid = enc_mask.to(torch.bool)
        else:
            mem_len = getattr(model.surrogate, "mem_len", max_len)
            mem_valid = torch.ones(1, mem_len, dtype=torch.bool, device=device)
        memory, _ = model.surrogate(z, mem_valid, causal_self=False) # (1, Lm, D)
    else:
        mem_len = getattr(model.surrogate, "mem_len", max_len) if getattr(model, "surrogate", None) else max_len
        memory = torch.zeros(1, mem_len, model.dec_emb.embedding_dim, device=device)
        mem_valid = torch.ones(1, mem_len, dtype=torch.bool, device=device)

    # TransformerDecoder: memory_key_padding_mask에서 True==pad(무시) 이므로 뒤집기
    mem_pad_mask = ~mem_valid[:, :memory.size(1)]
    # ───────────────────────────────────────────────────────────

    # AR 생성 루프 (EOS 미사용, PAD로만 종료)
    with torch.no_grad():
        for t in range(1, max_len):
            tok_emb = model.dec_emb(generated[:, :t])        # (1, t, D)
            pos_emb = model.dec_pos[:, :t, :]                # (1, t, D)
            z_emb  = model.latent2emb(z).unsqueeze(1).expand(-1, t, -1)
            tgt = tok_emb + pos_emb + z_emb                  # (1, t, D)

            dec_out = model.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=full_tgt_mask[:t, :t],
                memory_key_padding_mask=mem_pad_mask,
            )
            # (T,B,D) 대비 가드 → 항상 (B,T,D)
            if dec_out.dim() == 3 and dec_out.size(0) != generated.size(0):
                dec_out = dec_out.transpose(0, 1)

            step_logits = model.out(dec_out)[:, -1, :]       # (B,V)

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
                    vals, idxs = torch.topk(step_logits, k, dim=-1)  # idxs: 원 vocab id
                    masked = torch.full_like(step_logits, float("-inf"))
                    masked.scatter_(1, idxs, vals)                   # 원 vocab 공간 내 마스킹
                    step_logits = masked
                probs = torch.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated[:, t] = next_token

            # PAD 나오면 종료(원 동작 유지)
            if (next_token == tokenizer.pad_idx).all():
                break

    # 시퀀스 추출 (EOS 미트림)
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
