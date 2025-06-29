import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from vae_module import Config, Tokenizer, load_vae, encode_batch, pad_collate


class MLPRegressor(torch.nn.Module):
    def __init__(self, in_dim: int):
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


def read_sequences(df: pd.DataFrame):
    if "sequence" in df.columns:
        return df["sequence"].astype(str).tolist()
    if "mutated_sequence" in df.columns:
        return df["mutated_sequence"].astype(str).tolist()
    raise ValueError("No sequence column found")


def train_regressor(z: torch.Tensor, y: torch.Tensor, device: torch.device):
    scaler = StandardScaler()
    z_scaled = torch.tensor(scaler.fit_transform(z.cpu().numpy()), dtype=torch.float32)

    model = MLPRegressor(z_scaled.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2.29e-4, weight_decay=9.17e-5)
    loss_fn = torch.nn.MSELoss()

    dataset = TensorDataset(z_scaled.to(device), y.to(device))
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


def score_assay(csv_path: Path, cfg: Config, tokenizer: Tokenizer, vae) -> float:
    df = pd.read_csv(csv_path)
    seqs = read_sequences(df)
    y = torch.tensor(df["DMS_score"].astype(float).values, dtype=torch.float32)

    dataset = TensorDataset(torch.arange(len(seqs)))
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=lambda idx: pad_collate([tokenizer.to_tensor(seqs[i.item()]) for i in idx], tokenizer.pad_idx),
    )
    with torch.no_grad():
        z = encode_batch(vae, loader, tokenizer)

    mlp, scaler = train_regressor(z, y, cfg.device)
    mlp.eval()
    z_scaled = torch.tensor(scaler.transform(z.cpu().numpy()), dtype=torch.float32).to(cfg.device)
    with torch.no_grad():
        preds = mlp(z_scaled).cpu().numpy()

    rho = spearmanr(preds, y.numpy()).correlation
    return rho


def main():
    p = argparse.ArgumentParser(description="Score ProteinGym assay with ESMS-VAE")
    p.add_argument("--checkpoint", required=True, help="Path to ESMS-VAE checkpoint")
    p.add_argument("--assay", required=True, help="CSV file for one DMS assay")
    args = p.parse_args()

    cfg = Config(model_path=args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_esm()
    vae = load_vae(cfg, len(tokenizer.vocab), tokenizer.pad_idx, tokenizer.bos_idx)
    if cfg.device == "cuda" and torch.cuda.device_count() > 1:
        vae = torch.nn.DataParallel(vae)

    rho = score_assay(Path(args.assay), cfg, tokenizer, vae)
    print(f"{Path(args.assay).stem}\tSpearman: {rho:.4f}")


if __name__ == "__main__":
    main()
