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


class VAEWithSurrogate(nn.Module):
    """Wrapper bundling a VAE and a surrogate network."""

    def __init__(
        self,
        vae: VAETransformerDecoder,
        surrogate: Optional[Z2MemorySurrogate] = None,
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.vae(x, mask)
