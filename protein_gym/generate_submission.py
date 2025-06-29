"""Generate ProteinGym leaderboard submission files using ESMS VAE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


from vae_module import (
    Config,
    Tokenizer,
    load_vae,
    encode_batch,
    pad_collate,
)


class MLPRegressor(torch.nn.Module):
    """Simple feed-forward network used for DMS score prediction."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def read_sequences(df: pd.DataFrame) -> List[str]:
    if "sequence" in df.columns:
        return df["sequence"].astype(str).tolist()
    if "mutated_sequence" in df.columns:
        return df["mutated_sequence"].astype(str).tolist()
    raise ValueError("No sequence column found")


def train_mlp(z: torch.Tensor, y: torch.Tensor, device: torch.device) -> MLPRegressor:
    scaler = StandardScaler()
    z_scaled = torch.tensor(scaler.fit_transform(z.cpu().numpy()), dtype=torch.float32)
    X_train = z_scaled.to(device)
    y_train = y.to(device)

    torch.manual_seed(42)
    model = MLPRegressor(X_train.size(1)).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=2.29e-4, weight_decay=9.17e-5
    )
    loss_fn = torch.nn.MSELoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for _ in range(300):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model, scaler


def process_file(path: Path, cfg: Config, tokenizer: Tokenizer, model) -> pd.DataFrame:
    df = pd.read_csv(path)
    seqs = read_sequences(df)
    if "DMS_score" in df.columns:
        y = torch.tensor(df["DMS_score"].astype(float).values, dtype=torch.float32)
    else:
        raise ValueError("DMS_score column is required")

    dataset = TensorDataset(torch.arange(len(seqs)))  # placeholder indices
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=lambda idx: pad_collate([tokenizer.to_tensor(seqs[i.item()]) for i in idx], tokenizer.pad_idx),
    )
    with torch.no_grad():
        z = encode_batch(model, loader, tokenizer)

    mlp, scaler = train_mlp(z, y, device=cfg.device)
    mlp.eval()
    z_scaled = torch.tensor(
        scaler.transform(z.cpu().numpy()), dtype=torch.float32
    ).to(cfg.device)
    with torch.no_grad():
        preds = mlp(z_scaled).cpu().numpy()

    rho = spearmanr(preds, y.numpy()).correlation
    print(f"{path.name}: Spearman rho={rho:.4f}")

    out = pd.DataFrame({"sequence": seqs, "prediction": preds})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate ProteinGym submission")
    p.add_argument("--data-dir", required=True, help="Directory with ProteinGym CSV files")
    p.add_argument("--output-dir", required=True, help="Where to store prediction files")
    p.add_argument("--weights", default="models/vae_epoch380.pt", help="Path to pretrained VAE weights")
    args = p.parse_args()

    cfg = Config(model_path=args.weights, device="cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_esm()
    vae = load_vae(cfg, len(tokenizer.vocab), tokenizer.pad_idx, tokenizer.bos_idx)
    if cfg.device == "cuda" and torch.cuda.device_count() > 1:
        vae = torch.nn.DataParallel(vae)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(Path(args.data_dir).glob("*.csv")):
        pred_df = process_file(csv_path, cfg, tokenizer, vae)
        out_file = Path(args.output_dir) / f"{csv_path.stem}_pred.csv"
        pred_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
