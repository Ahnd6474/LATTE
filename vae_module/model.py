import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

DROPOUT = 0.1
LATENT_DIM = 256
EMB_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
FFN_DIM = 512
MAX_LEN = 512


class CrossDiagBias(nn.Module):
    """Gaussian alignment bias with optional hard band masking."""

    def __init__(
        self,
        init_alpha: float = 0.05,
        a_span: float = 0.5,
        d_span: float = 0.15,
        band_W_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.a_raw = nn.Parameter(torch.zeros(1))
        self.d_raw = nn.Parameter(torch.zeros(1))
        self.a_span = float(a_span)
        self.d_span = float(d_span)
        self.band_W_tokens = int(band_W_tokens)

    @torch.no_grad()
    def clamp_alpha_(self, min_alpha: float = 1e-6, max_alpha: float = 1.0) -> None:
        """Clamp the learned ``alpha`` parameter to keep the bias stable."""

        self.alpha.clamp_(min_alpha, max_alpha)

    def forward(self, T: int, M: int, device: torch.device) -> torch.Tensor:
        """Return an additive bias mask shaped ``(T, M)`` for cross attention."""

        t = torch.linspace(0.0, 1.0, T, device=device)[:, None]
        i = torch.linspace(0.0, 1.0, M, device=device)[None, :]

        a = 1.0 + self.a_span * torch.tanh(self.a_raw)
        d = self.d_span * torch.tanh(self.d_raw)

        m_hat_norm = (a * t + d).clamp(0.0, 1.0)
        m_hat_idx = m_hat_norm * (M - 1)

        alpha = self.alpha.clamp_min(1e-6)
        i_idx = torch.arange(M, device=device, dtype=m_hat_idx.dtype)[None, :]
        bias = -alpha * (i_idx - m_hat_idx).pow(2)

        if self.band_W_tokens > 0:
            W = max(1, int(self.band_W_tokens * M / max(T, 1)))
            band = (i_idx - m_hat_idx).abs() <= W
            bias = bias.masked_fill(~band, float("-inf"))

        return bias


class SmallTransformer(nn.Module):
    """Simple Transformer encoder used in the notebook."""

    def __init__(self, vocab_size: int, emb_dim: int, layers: int, heads: int,
                 ffn_dim: int, max_len: int, pad_idx: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=DROPOUT,
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_idx = self.emb.padding_idx if self.emb.padding_idx is not None else 0
        mask = x != pad_idx
        h = self.emb(x) + self.pos[:, : x.size(1), :]
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.ln(h), mask


class VAETransformerDecoder(nn.Module):
    """VAE model from the notebook."""

    def __init__(self, encoder: SmallTransformer, vocab_size: int,
                 latent_dim: int = LATENT_DIM, emb_dim: int = EMB_DIM,
                 num_layers: int = NUM_LAYERS, num_heads: int = NUM_HEADS,
                 ffn_dim: int = FFN_DIM, max_len: int = MAX_LEN,
                 pad_token: int = 0, bos_token: int = 1):
        super().__init__()
        self.encoder = encoder
        self.pad_token = pad_token
        self.bos_token = bos_token

        self.to_mu = nn.Linear(emb_dim, latent_dim)
        self.to_logvar = nn.Linear(emb_dim, latent_dim)
        self.latent2emb = nn.Linear(latent_dim, emb_dim)

        self.dec_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token)
        self.dec_pos = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        h_enc, enc_mask = self.encoder(x)
        denom = enc_mask.sum(1, keepdim=True).clamp_min(1)
        pooled = (h_enc * enc_mask.unsqueeze(-1)).sum(1) / denom
        mu, logvar = self.to_mu(pooled), self.to_logvar(pooled)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        B, L = x.size()
        dec_in = torch.full((B, L), self.bos_token, device=x.device, dtype=torch.long)
        dec_in[:, 1:] = x[:, :-1]
        emb = self.dec_emb(dec_in) + self.dec_pos[:, :L, :]
        z_emb = self.latent2emb(z).unsqueeze(1).expand(-1, L, -1)
        emb = emb + z_emb

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h_dec = self.decoder(
            tgt=emb,
            memory=h_enc,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=~mask,
            memory_key_padding_mask=~enc_mask,
        )
        logits = self.out(h_dec)
        return logits, mu, logvar, h_enc, enc_mask


class Z2MemorySurrogate(nn.Module):
    """Small transformer that predicts decoder memory from latent ``z``."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        max_len: int,
        layers: int = 2,
        heads: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 3 * d_model
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.z_ln = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self, z: torch.Tensor, mask_bool: torch.Tensor, causal_self: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = mask_bool.shape
        base = self.token.expand(B, L, -1) + self.pos[:, :L, :]
        zemb = self.z_ln(self.z_proj(z)).unsqueeze(1).expand(-1, L, -1)
        h = base + zemb
        src_mask = None
        if causal_self:
            src_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=h.device), diagonal=1
            )
        h = self.enc(h, mask=src_mask, src_key_padding_mask=~mask_bool)
        return self.out_ln(h), mask_bool


# ============================================
# 2) VAEWithSurrogate 확장: 바이어스/LN/체크포인트
#    - 모델 state_dict에 같이 저장됩니다.
# ============================================
class VAEWithSurrogate(nn.Module):
    """Wrapper bundling a VAE and a surrogate network."""

    def __init__(
        self,
        vae: VAETransformerDecoder,
        surrogate: Optional[Z2MemorySurrogate] = None,
        use_sur_ln: bool = True,
        use_sur_gate: bool = True,
        use_diag_bias: bool = True,
        diag_init_alpha: float = 0.05,
        diag_a_span: float = 0.5,
        diag_d_span: float = 0.15,
        diag_band_W: int = 8,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.surrogate = surrogate

        for name in [
            "encoder",
            "decoder",
            "dec_emb",
            "dec_pos",
            "latent2emb",
            "pad_token",
            "bos_token",
            "out",
        ]:
            setattr(self, name, getattr(vae, name))

        d_model = self.dec_emb.embedding_dim
        self.sur_ln = nn.LayerNorm(d_model, elementwise_affine=True) if use_sur_ln else None
        self.use_sur_gate = bool(use_sur_gate)
        self.sur_gate = (
            nn.Parameter(torch.full((d_model,), math.log(0.3 / 0.7)))
            if use_sur_gate
            else None
        )

        self.diag_bias = (
            CrossDiagBias(
                init_alpha=diag_init_alpha,
                a_span=diag_a_span,
                d_span=diag_d_span,
                band_W_tokens=diag_band_W,
            )
            if use_diag_bias
            else None
        )

        self._z_cached: Optional[torch.Tensor] = None

    # -------------------------
    # Surrogate 메모리 구성 헬퍼
    # -------------------------
    @torch.no_grad()
    def build_surrogate_memory(
        self, z: torch.Tensor, x_gt_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return surrogate decoder memory and its padding mask."""

        device = next(self.parameters()).device
        z = z.to(device)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.size(0)

        if self.surrogate is None:
            M = MAX_LEN
            memory = torch.zeros(B, M, self.dec_emb.embedding_dim, device=device)
            mem_valid = torch.ones(B, M, dtype=torch.bool, device=device)
        else:
            if x_gt_ids is not None:
                if x_gt_ids.dim() == 1:
                    x_gt_ids = x_gt_ids.unsqueeze(0)
                x_gt_ids = x_gt_ids.to(device)
                _, enc_mask = self.encoder(x_gt_ids)
                mem_valid = enc_mask.to(torch.bool)
            else:
                M = int(self.surrogate.pos.size(1))
                mem_valid = torch.ones(B, M, dtype=torch.bool, device=device)
            memory, _ = self.surrogate(z, mem_valid, causal_self=False)

        if self.sur_ln is not None:
            memory = self.sur_ln(memory)
        if self.use_sur_gate and self.sur_gate is not None:
            gate = torch.sigmoid(self.sur_gate)
            memory = memory * gate

        mem_pad_mask = ~mem_valid
        return memory, mem_pad_mask

    # -------------------------
    # 디코딩 한 스텝 로짓(+가우시안 바이어스)
    # -------------------------
    def decode_step(self, prefix_ids: torch.Tensor,
                    memory: torch.Tensor, mem_pad_mask: torch.Tensor,
                    tokenizer=None, use_bias: bool = True) -> torch.Tensor:
        """
        입력: prefix_ids (B,T), memory (B,M,D), mem_pad_mask (B,M; True==PAD)
        출력: 마지막 스텝 로짓 (B,V)
        """
        device = prefix_ids.device
        B, T = prefix_ids.size()
        M = memory.size(1)

        tok = self.dec_emb(prefix_ids)
        pos = self.dec_pos[:, :T, :]
        if self._z_cached is None:
            raise RuntimeError(
                "self._z_cached가 없습니다. 학습/추론 루프에서 self._z_cached = z (B,D)로 세팅하세요."
            )
        z_emb = self.latent2emb(self._z_cached).unsqueeze(1).expand(-1, T, -1)
        tgt = tok + pos + z_emb

        tgt_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

        memory_mask = None
        if use_bias and (self.diag_bias is not None):
            memory_mask = self.diag_bias(T, M, device=device)

        h = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,                 # 가산형 로짓 바이어스
            memory_key_padding_mask=mem_pad_mask,
        )
        if h.dim() == 3 and h.size(0) != B:  # (T,B,D) → (B,T,D)
            h = h.transpose(0, 1)
        logits = self.out(h)[:, -1, :]       # (B,V)
        if tokenizer is not None:
            pad_idx = getattr(
                tokenizer, "pad_idx", getattr(tokenizer, "pad_token_id", self.pad_token)
            )
            if pad_idx is not None:
                logits[:, int(pad_idx)] = -float("inf")
        else:
            logits[:, int(self.pad_token)] = -float("inf")
        return logits

    # -------------------------
    # 체크포인트 저장/복구
    # -------------------------
    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                        epoch: Optional[int] = None, step: Optional[int] = None,
                        extra: Optional[dict] = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "model_state": self.state_dict(),      # ★ diag_bias/sur_ln 모두 포함
            "epoch": epoch,
            "step": step,
            "extra": extra or {},
        }
        if optimizer is not None:
            ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, path)

    @staticmethod
    def load_checkpoint(path: str, model: "VAEWithSurrogate",
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        map_location: Optional[str] = None, strict: bool = True) -> dict:
        ckpt = torch.load(path, map_location=map_location or "cpu")
        model.load_state_dict(ckpt["model_state"], strict=strict)
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt


def guided_alignment_kl(
    attn_probs: torch.Tensor, m_hat_idx: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Compute KL(attn || Gaussian) for alignment guidance during training."""

    B, H, T, M = attn_probs.shape
    device = attn_probs.device
    i_idx = torch.arange(M, device=device).float()[None, None, None, :]
    m = m_hat_idx[..., None]
    gauss = torch.exp(-((i_idx - m).pow(2)) / (2.0 * (sigma ** 2)))
    gauss = gauss / (gauss.sum(-1, keepdim=True) + 1e-9)
    kl = (attn_probs.clamp_min(1e-9).log() - gauss.clamp_min(1e-9).log()) * attn_probs
    return kl.sum(dim=-1).mean()
