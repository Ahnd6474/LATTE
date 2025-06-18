"""Predict protein melting temperatures using VAE embeddings and Gaussian Process regression."""

from typing import List, Tuple

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_squared_error

from vae_module import (
    Config,
    Tokenizer,
    load_vae,
    SequenceDataset,
    pad_collate,
    encode_batch,
)



def load_tm_dataset(
    csv_path: str, tokenizer: Tokenizer, max_len: int
) -> Tuple[List[str], np.ndarray]:
    """Load sequences and Tm values from a CSV file.


    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sequence", "Tm"])

    sequences: List[str] = []
    tm_values: List[float] = []
    for seq, tm in zip(df["sequence"], df["Tm"]):
        s = str(seq)
        if not s or s.lower() == "nan":
            continue

        if len(s) > max_len:
            continue
        if any(c not in tokenizer.vocab for c in s):
            continue
        sequences.append(s)
        tm_values.append(float(tm))

    return sequences, np.array(tm_values, dtype=float)

def load_tm_dataset(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """Load sequences and Tm values from a CSV file."""
    df = pd.read_csv(csv_path)
    seqs = df["sequence"].astype(str).tolist()
    tm = df["Tm"].astype(float).values
    return seqs, tm



def encode_sequences(
    sequences: List[str], cfg: Config, tokenizer: Tokenizer, model
) -> np.ndarray:
    """Encode sequences into latent vectors using the VAE."""
    trimmed = [s[: cfg.max_len] for s in sequences]
    dataset = SequenceDataset(trimmed, tokenizer, cfg.max_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_idx),
    )
    with torch.no_grad():
        z = encode_batch(model, loader, tokenizer)
    return z.cpu().numpy()


def main(
    csv_path: str = "tm_datasets.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    cache: str | None = None,
) -> None:
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

    if cache and os.path.exists(cache):
        data = np.load(cache)
        Z = data["Z"]
        tm = data["tm"]
    else:
        sequences, tm = load_tm_dataset(csv_path, tokenizer, cfg.max_len)
        sequences, tm = load_tm_dataset(csv_path, tokenizer, cfg.max_len)
        sequences, tm = load_tm_dataset(csv_path)

        Z = encode_sequences(sequences, cfg, tokenizer, model)
        if cache:
            np.savez(cache, Z=Z, tm=tm)

    X_train, X_test, y_train, y_test = train_test_split(
        Z,
        tm,
        test_size=test_size,
        random_state=random_state,
    )

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=3,
        alpha=1e-6,
    )
    gp.fit(X_train, y_train)

    preds = gp.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
