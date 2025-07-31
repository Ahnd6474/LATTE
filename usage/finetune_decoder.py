from typing import List
import os
import random

import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from vae_module import (
    Config,
    Tokenizer,
    load_vae,
    SequenceDataset,
    pad_collate,
    decode_batch,
)

# ---- Configuration ----

# Update these paths and settings as needed before running the script.
DATA_PATH = "/kaggle/input/uniref50-sub/uniref50_subsample.fasta"
CHECKPOINT_PATH = "models/vae_epoch380.pt"
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 64
NOISE_PROB = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/kaggle/working/finetuned"
MAX_LEN = 512


def read_fasta(path: str) -> List[str]:
    sequences: List[str] = []
    with open(path) as fh:
        seq = ""
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences


def random_sample(seqs: List[str], k: int) -> List[str]:
    """Return a random subset of sequences of size ``k``."""
    if len(seqs) <= k:
        return list(seqs)
    return random.sample(seqs, k)


def add_noise(tokens: torch.Tensor, prob: float, vocab_size: int, pad_idx: int) -> torch.Tensor:
    if prob <= 0.0:
        return tokens
    noise_mask = (torch.rand(tokens.size(), device=tokens.device) < prob) & (tokens != pad_idx)
    random_tokens = torch.randint(0, vocab_size, tokens.size(), device=tokens.device)
    out = tokens.clone()
    out[noise_mask] = random_tokens[noise_mask]
    return out


def forward_noisy(model, x, mask, noise_prob: float, vocab_size: int):
    with torch.no_grad():
        h_enc, enc_mask = model.encoder(x)
    pooled = (h_enc * enc_mask.unsqueeze(-1)).sum(1) / enc_mask.sum(1, True)
    mu = model.to_mu(pooled)
    logvar = model.to_logvar(pooled)
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    B, L = x.size()
    dec_in = torch.full((B, L), model.bos_token, device=x.device, dtype=torch.long)
    dec_in[:, 1:] = x[:, :-1]
    dec_in = add_noise(dec_in, noise_prob, vocab_size, model.pad_token)

    emb = model.dec_emb(dec_in) + model.dec_pos[:, :L, :]
    z_emb = model.latent2emb(z).unsqueeze(1).expand(-1, L, -1)
    emb = emb + z_emb

    tgt_mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
    h_dec = model.decoder(
        tgt=emb,
        memory=h_enc,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=~mask,
        memory_key_padding_mask=~enc_mask,
    )
    logits = model.out(h_dec)
    return logits, mu, logvar


def _split_tokens(seq: str) -> List[str]:
    """Split a generated sequence string back into tokens."""
    tokens: List[str] = []
    i = 0
    while i < len(seq):
        if seq[i] == "<":
            j = seq.find(">", i)
            if j == -1:
                break
            tokens.append(seq[i : j + 1])
            i = j + 1
        else:
            tokens.append(seq[i])
            i += 1
    return tokens


def tokens_to_tensor(tokens: List[str], tokenizer: Tokenizer, max_len: int) -> torch.Tensor:
    ids = [tokenizer.get_idx(tok) for tok in tokens if tok in tokenizer.tok_to_idx]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)


def evaluate(model, loader, tokenizer, max_len: int) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            mask = x != tokenizer.pad_idx
            h_enc, enc_mask = model.encoder(x)
            pooled = (h_enc * enc_mask.unsqueeze(-1)).sum(1) / enc_mask.sum(1, True)
            mu = model.to_mu(pooled)
            z = mu
            sequences = decode_batch(
                model,
                z,
                tokenizer,
                max_len,
                truncate_lens=mask.sum(1).tolist(),
            )
            for seq_pred, xt, m in zip(sequences, x, mask):
                tokens = _split_tokens(seq_pred)
                if tokens and tokens[0] == tokenizer.bos_token:
                    tokens = tokens[1:]
                pred_ids = tokens_to_tensor(tokens, tokenizer, max_len).to(device)
                L = int(m.sum().item())
                correct += (pred_ids[:L] == xt[:L]).sum().item()
                total += L
    return correct / total * 100 if total > 0 else 0.0


def main() -> None:
    tokenizer = Tokenizer.from_esm()
    cfg = Config(model_path=CHECKPOINT_PATH, device=DEVICE, max_len=MAX_LEN)
    model = load_vae(
        cfg,
        vocab_size=len(tokenizer.vocab),
        pad_idx=tokenizer.pad_idx,
        bos_idx=tokenizer.bos_idx,
    )

    for p in model.encoder.parameters():
        p.requires_grad = False

    sequences = random_sample(read_fasta(DATA_PATH), 300000)
    dataset = SequenceDataset(sequences, tokenizer, MAX_LEN)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_idx),
    )

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    device = torch.device(DEVICE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(device)
            mask = x != tokenizer.pad_idx
            logits, mu, logvar = forward_noisy(model, x, mask, NOISE_PROB, len(tokenizer.vocab))
            ce = cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), ignore_index=tokenizer.pad_idx)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = ce + kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, val_loader, tokenizer, MAX_LEN)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_acc={acc:.2f}%")
        output_path = f"{OUTPUT_DIR}/decoder_ft_epoch{epoch}.pt"
        torch.save({"model_sd": model.state_dict()}, output_path)
        print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
