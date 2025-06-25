import argparse
import os
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from vae_module import Config, Tokenizer, load_vae
from amino_acid_pca import load_sequences, encode_sequences


def main(
    directory: str = "amino acids",
    max_per_class: int = 100,
    neighbors: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    cache: str | None = None,
) -> None:
    """Train and evaluate a KNN classifier on amino acid sequences."""
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
        data = np.load(cache, allow_pickle=True)
        Z = data["Z"]
        labels = data["labels"].tolist()
    else:
        labels, sequences = load_sequences(directory, tokenizer, max_per_class=max_per_class)
        Z = encode_sequences(sequences, cfg, tokenizer, model).cpu().numpy()
        if cache:
            np.savez(cache, Z=Z, labels=np.array(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        Z,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN on amino acid sequences")
    parser.add_argument("--directory", default="amino acids", help="Directory with CSV files")
    parser.add_argument("--max-per-class", type=int, default=100, help="Max sequences to load per class")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--cache", help="Path to .npz file for saving/loading latent vectors")
    args = parser.parse_args()
    main(
        directory=args.directory,
        max_per_class=args.max_per_class,
        neighbors=args.neighbors,
        test_size=args.test_size,
        cache=args.cache,
    )
