"""Example script demonstrating the vae_module API."""

from torch.utils.data import DataLoader

from vae_module import (
    Config,
    Tokenizer,
    load_vae,
    encode,
    decode,
    encode_batch,
    SequenceDataset,
    pad_collate,
)


def main() -> None:
    """Load the pretrained model and encode/decode example sequences."""
    cfg = Config(model_path="models/vae_epoch380.pt")
    tokenizer = Tokenizer.from_esm()
    model = load_vae(
        cfg,
        vocab_size=len(tokenizer.vocab),
        pad_idx=tokenizer.pad_idx,
        bos_idx=tokenizer.bos_idx,
    )

    sequence = "MKTFFVLLL"
    z = encode(model, sequence, tokenizer, cfg.max_len)
    reconstructed = decode(model, z, tokenizer, cfg.max_len, truncate_len=len(sequence))

    print("Original sequence:     ", sequence)
    print("Latent vector shape:   ", tuple(z.shape))
    print("Reconstructed sequence:", reconstructed)

    # Batch encoding example
    sequences = [sequence, "ACDEFGHIKLMNPQRSTVWY"]
    dataset = SequenceDataset(sequences, tokenizer, cfg.max_len)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda batch: pad_collate(batch, tokenizer.pad_idx),
    )
    Z = encode_batch(model, loader, tokenizer)
    print("Batch latent tensor shape:", tuple(Z.shape))


if __name__ == "__main__":
    main()
