from typing import List

import torch
from tqdm import trange

from .logger import setup_logger
from .model import VAETransformerDecoder
from .classes import Tokenizer
from .utils import tensor_to_sequence

logger = setup_logger(__name__)


def decode(model: VAETransformerDecoder, z: torch.Tensor, tokenizer: Tokenizer, max_len: int) -> str:
    """Decode a latent vector into a sequence."""
    model.eval()
    device = next(model.parameters()).device
    z = z.unsqueeze(0).to(device)

    generated = torch.full((1, max_len), tokenizer.pad_idx, device=device, dtype=torch.long)
    generated[:, 0] = tokenizer.bos_idx

    tgt_mask = torch.triu(torch.full((max_len, max_len), float("-inf"), device=device), diagonal=1)
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
    return tensor_to_sequence(ids, tokenizer)


def decode_batch(model: VAETransformerDecoder, Z: torch.Tensor, tokenizer: Tokenizer, max_len: int) -> List[str]:
    """Decode a batch of latent vectors."""
    seqs = [decode(model, z, tokenizer, max_len) for z in Z]
    return seqs
