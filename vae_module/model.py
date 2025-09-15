from typing import Tuple, Optional

import torch
import torch.nn as nn

DROPOUT = 0.1
LATENT_DIM = 256
EMB_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
FFN_DIM = 512
MAX_LEN = 512

# ===============================
# 1) 가우시안 정렬 바이어스 모듈
# ===============================
import os
import torch
import torch.nn as nn

class CrossDiagBias(nn.Module):
    """i ≈ a*t + δ 근방을 선호하도록 cross-attn 로짓에 -α*(i-(a t+δ))^2 가산."""
    def __init__(self, max_mem_len: int, init_alpha: float = 0.05, init_a: float = 1.0, init_delta: float = 0.0):
        super().__init__()
        self.max_mem = int(max_mem_len)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))  # > 0
        self.a     = nn.Parameter(torch.tensor(float(init_a)))
        self.delta = nn.Parameter(torch.tensor(float(init_delta)))

    @torch.no_grad()
    def clamp_(self, min_alpha: float = 1e-6, max_alpha: float = 1.0):
        self.alpha.clamp_(min_alpha, max_alpha)

    def forward(self, t: int, device: torch.device) -> torch.Tensor:
        """(t, M) 가산형 로짓 바이어스 반환 (PyTorch TransformerDecoder에 memory_mask로 전달 가능)."""
        tt = torch.arange(t, device=device).unsqueeze(1).float()             # (t,1)
        mm = torch.arange(self.max_mem, device=device).unsqueeze(0).float()  # (1,M)
        center = self.a * tt + self.delta
        dist2  = (center - mm) ** 2
        alpha  = torch.clamp(self.alpha, 1e-6, 1.0)
        return -alpha * dist2  # (t,M)


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
        mask = x != self.emb.padding_idx
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
        pooled = (h_enc * enc_mask.unsqueeze(-1)).sum(1) / enc_mask.sum(1, True)
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
        use_diag_bias: bool = True,
        diag_init_alpha: float = 0.05,
        diag_init_a: float = 1.0,
        diag_init_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.surrogate = surrogate

        # 기존 편의 alias (원 코드 유지)
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

        # (선택) surrogate 출력 뒤 LayerNorm — γ/β도 state_dict에 포함됨
        self.sur_ln = nn.LayerNorm(self.dec_emb.embedding_dim, elementwise_affine=True) if use_sur_ln else None

        # (선택) 가우시안 정렬 바이어스 — 파라미터(α,a,δ) state_dict에 포함됨
        if use_diag_bias:
            mem_len = MAX_LEN
            if surrogate is not None and hasattr(surrogate, "pos"):
                try:
                    mem_len = int(surrogate.pos.size(1))
                except Exception:
                    pass
            self.diag_bias = CrossDiagBias(mem_len, diag_init_alpha, diag_init_a, diag_init_delta)
        else:
            self.diag_bias = None

    # -------------------------
    # Surrogate 메모리 구성 헬퍼
    # -------------------------
    @torch.no_grad()
    def build_surrogate_memory(self, z: torch.Tensor, x_gt_ids: Optional[torch.Tensor] = None):
        """
        반환: memory (1,M,D), mem_pad_mask (1,M)  (True==PAD)
        """
        device = next(self.parameters()).device
        z = z.flatten().unsqueeze(0).to(device)
        if self.surrogate is None:
            M = MAX_LEN
            memory = torch.zeros(1, M, self.dec_emb.embedding_dim, device=device)
            mem_valid = torch.ones(1, M, dtype=torch.bool, device=device)
        else:
            if x_gt_ids is not None:
                if x_gt_ids.dim() == 1:
                    x_gt_ids = x_gt_ids.unsqueeze(0)
                x_gt_ids = x_gt_ids.to(device)
                _, enc_mask = self.encoder(x_gt_ids)  # True==valid
                mem_valid = enc_mask.to(torch.bool)
            else:
                M = int(self.surrogate.pos.size(1))
                mem_valid = torch.ones(1, M, dtype=torch.bool, device=device)
            memory, _ = self.surrogate(z, mem_valid, causal_self=False)

        # (선택) surrogate 출력 정규화
        if self.sur_ln is not None:
            memory = self.sur_ln(memory)

        mem_pad_mask = ~mem_valid  # True==PAD(무시)
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

        tok = self.dec_emb(prefix_ids)                 # (B,T,D)
        pos = self.dec_pos[:, :T, :]                   # (1,T,D)
        if not hasattr(self, "_z_cached"):
            raise RuntimeError("self._z_cached가 없습니다. 학습/추론 루프에서 self._z_cached = z (B,D)로 세팅하세요.")
        z_emb = self.latent2emb(self._z_cached).unsqueeze(1).expand(-1, T, -1)
        tgt = tok + pos + z_emb

        tgt_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

        memory_mask = None
        if use_bias and (self.diag_bias is not None):
            memory_mask = self.diag_bias(int(T), device=device)  # (T,M)

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
        if tokenizer is not None and hasattr(tokenizer, "pad_idx"):
            logits[:, tokenizer.pad_token] = -float("inf") if hasattr(tokenizer, "pad_token") else -float("inf")
            logits[:, tokenizer.pad_idx] = -float("inf")
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
