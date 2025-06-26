"""Demonstrate encoding long sequences with encode_long_batch."""

from vae_module import (
    Config,
    Tokenizer,
    load_vae,
    encode_long_batch,
)


def main() -> None:
    """Load the pretrained model and encode a list of long sequences."""
    cfg = Config(model_path="models/vae_epoch380.pt")
    tokenizer = Tokenizer.from_esm()
    model = load_vae(
        cfg,
        vocab_size=len(tokenizer.vocab),
        pad_idx=tokenizer.pad_idx,
        bos_idx=tokenizer.bos_idx,
    )

    sequences = [
        "MKTFFVLLL" * 100,
        "ACDEFGHIKLMNPQRSTVWY" * 50,
    ]
    zs = encode_long_batch(model, sequences, tokenizer, cfg.max_len, overlap=256)

    for i, z_stack in enumerate(zs, 1):
        print(f"Sequence {i} latent stack shape: {tuple(z_stack.shape)}")


if __name__ == "__main__":
    main()
