import glob
import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from vae_module import (
    Config,
    Tokenizer,
    load_vae,
    SequenceDataset,
    pad_collate,
    encode_batch,
)


def load_sequences(
    directory: str,
    tokenizer: Tokenizer,
    max_per_class: int | None = None,
) -> Tuple[List[str], List[str]]:
    """Load sequences from CSV files and drop those with invalid tokens."""
    labels: List[str] = []
    sequences: List[str] = []
    for csv_path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        label = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        seqs = df["Sequence"].tolist()
        if max_per_class is not None:
            seqs = seqs[:max_per_class]
        for seq in seqs:
            if not isinstance(seq, str):
                continue
            seq = seq.strip()
            if all(c in tokenizer.vocab for c in seq):
                sequences.append(seq)
                labels.append(label)
    return labels, sequences


def encode_sequences(
    sequences: List[str],
    cfg: Config,
    tokenizer: Tokenizer,
    model,
    *,
    truncate: bool = True,
) -> torch.Tensor:
    """Encode sequences into latent vectors using the VAE model.

    Parameters
    ----------
    sequences:
        List of raw amino-acid sequences.
    cfg:
        Configuration object specifying ``max_len`` and ``batch_size``.
    tokenizer:
        Tokenizer instance matching the model vocabulary.
    model:
        Loaded VAE model.
    truncate:
        If ``True`` (default), sequences longer than ``cfg.max_len`` are
        truncated; otherwise such sequences will raise
        :class:`SequenceLengthError` during tensorization.
    """

    if truncate:
        sequences = [s[: cfg.max_len] for s in sequences]

    dataset = SequenceDataset(sequences, tokenizer, cfg.max_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_idx),
    )
    return encode_batch(model, loader, tokenizer)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(model_path="models/vae_epoch380.pt", device=device)
    tokenizer = Tokenizer.from_esm()
    model = load_vae(
        cfg,
        vocab_size=len(tokenizer.vocab),
        pad_idx=tokenizer.pad_idx,
        bos_idx=tokenizer.bos_idx,
    )
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    labels, sequences = load_sequences("amino acids", tokenizer)
    Z = encode_sequences(sequences, cfg, tokenizer, model).cpu().numpy()

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z)

    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("hsv", len(unique_labels))
    color_map = {lab: palette[i] for i, lab in enumerate(unique_labels)}
    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(Z2[idx, 0], Z2[idx, 1], label=lab, color=color_map[lab], s=10, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
