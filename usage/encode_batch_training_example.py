"""Train MLP on embeddings obtained via encode_batch."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from vae_module import Config, Tokenizer, load_vae
from amino_acid_pca import load_sequences, encode_sequences


def main() -> None:
    """Encode sequences and train a simple MLP regressor."""
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

    # Load a subset of amino acid sequences
    _, sequences = load_sequences("amino acids", tokenizer, max_per_class=200)
    Z = encode_sequences(sequences, cfg, tokenizer, model).cpu().numpy()

    # Example regression targets: sequence lengths normalized to [0,1]
    y = np.array([len(s) / cfg.max_len for s in sequences], dtype=np.float32)

    X_tensor = torch.from_numpy(Z)
    y_tensor = torch.from_numpy(y).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    class MLPRegressor(nn.Module):
        def __init__(self, in_dim: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    reg = MLPRegressor(in_dim=Z.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(reg.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        reg.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = reg(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        reg.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = reg(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch:>2}/{num_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
