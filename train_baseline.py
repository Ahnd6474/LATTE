import argparse
from typing import List

import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae_module import (
    Tokenizer,
    SequenceDataset,
    pad_collate,
    SmallTransformer,
    VAETransformerDecoder,
    EMB_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    FFN_DIM,
    MAX_LEN,
)


def read_fasta(path: str) -> List[str]:
    """Read sequences from a FASTA or plain text file."""
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


def build_model(tokenizer: Tokenizer, device: torch.device) -> VAETransformerDecoder:
    encoder = SmallTransformer(
        vocab_size=len(tokenizer.vocab),
        emb_dim=EMB_DIM,
        layers=NUM_LAYERS,
        heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        max_len=MAX_LEN,
        pad_idx=tokenizer.pad_idx,
    ).to(device)
    return VAETransformerDecoder(
        encoder=encoder,
        vocab_size=len(tokenizer.vocab),
        pad_token=tokenizer.pad_idx,
        bos_token=tokenizer.bos_idx,
    ).to(device)


def train(
    sequences: List[str],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    save_path: str,
    max_len: int,
) -> None:
    tokenizer = Tokenizer.from_esm()
    model = build_model(tokenizer, device)
    dataset = SequenceDataset(sequences, tokenizer, max_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_idx),
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x in tqdm(loader, desc=f"Epoch {epoch}"):
            x = x.to(device)
            mask = x != tokenizer.pad_idx
            optimizer.zero_grad()
            logits, mu, logvar, *_ = model(x, mask)
            ce = cross_entropy(
                logits.view(-1, logits.size(-1)),
                x.view(-1),
                ignore_index=tokenizer.pad_idx,
            )
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = ce + kl
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        print(f"Epoch {epoch}: loss={avg:.4f}")

    torch.save({"model_sd": model.state_dict()}, save_path)
    print(f"Saved model to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline VAE without ESMS loss")
    parser.add_argument("data", help="Path to FASTA or text file with sequences")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="vae_baseline.pt")
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    seqs = read_fasta(args.data)
    train(
        seqs,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        save_path=args.output,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
