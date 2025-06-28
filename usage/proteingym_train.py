import argparse
from typing import List

import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

from vae_module import (
    Tokenizer,
    SequenceDataset,
    pad_collate,
    SmallTransformer,
    VAETransformerDecoder,
)


class LongSequenceDataset(Dataset):
    """Dataset for sequences longer than max_len using sliding windows."""

    def __init__(self, sequences: List[str], tokenizer: Tokenizer, max_len: int, overlap: int = 256) -> None:
        self.segments: List[torch.Tensor] = []
        step = max_len - overlap
        for seq in sequences:
            for start in range(0, len(seq), step):
                end = start + max_len
                part = seq[start:end]
                t = tokenizer.to_tensor(part)
                if t.size(0) < max_len:
                    pad_len = max_len - t.size(0)
                    t = torch.nn.functional.pad(t, (0, pad_len), value=tokenizer.pad_idx)
                    self.segments.append(t)
                    break
                self.segments.append(t)
                if end >= len(seq):
                    break

        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.segments[idx]


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


def build_model(tokenizer: Tokenizer, device: torch.device) -> VAETransformerDecoder:
    encoder = SmallTransformer(
        vocab_size=len(tokenizer.vocab),
        emb_dim=256,
        layers=4,
        heads=8,
        ffn_dim=512,
        max_len=tokenizer.max_len,
        pad_idx=tokenizer.pad_idx,
    ).to(device)
    return VAETransformerDecoder(
        encoder=encoder,
        vocab_size=len(tokenizer.vocab),
        pad_token=tokenizer.pad_idx,
        bos_token=tokenizer.bos_idx,
    ).to(device)


def evaluate(model: VAETransformerDecoder, loader: DataLoader, tokenizer: Tokenizer) -> float:
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            mask = x != tokenizer.pad_idx
            logits, _, _, _, _ = model(x, mask)
            preds = logits.argmax(-1)
            correct += ((preds == x) & mask).sum().item()
            total += mask.sum().item()
    return correct / total * 100 if total > 0 else 0.0


def train(
    sequences: List[str],
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    save_path: str,
    max_len: int,
    overlap: int,
) -> None:
    tokenizer = Tokenizer.from_esm(max_len=max_len)
    short = [s for s in sequences if len(s) <= max_len]
    long = [s for s in sequences if len(s) > max_len]
    short_ds = SequenceDataset(short, tokenizer, max_len)
    long_ds = LongSequenceDataset(long, tokenizer, max_len, overlap)
    dataset = ConcatDataset([short_ds, long_ds])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_idx),
    )
    model = build_model(tokenizer, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
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
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        acc = evaluate(model, loader, tokenizer)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} accuracy={acc:.2f}%")

    torch.save({"model_sd": model.state_dict()}, save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE for ProteinGym with long-sequence support")
    parser.add_argument("data", help="Path to FASTA file with sequences")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="vae_proteingym.pt")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=256)
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
        overlap=args.overlap,
    )

