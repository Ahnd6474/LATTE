import torch

from .config import Config
from .exceptions import DeviceNotAvailableError
from .logger import setup_logger
from .model import (
    SmallTransformer,
    VAETransformerDecoder,
    Z2MemorySurrogate,
    VAEWithSurrogate,
    EMB_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    FFN_DIM,
    MAX_LEN,
    LATENT_DIM,
    DROPOUT,
)

logger = setup_logger(__name__)


def load_vae(
    cfg: Config, vocab_size: int, pad_idx: int, bos_idx: int
) -> VAEWithSurrogate:
    """Load VAE (optionally with surrogate) from checkpoint defined in ``Config``."""

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceNotAvailableError(cfg.device)

    enc = SmallTransformer(
        vocab_size,
        EMB_DIM,
        NUM_LAYERS,
        NUM_HEADS,
        FFN_DIM,
        MAX_LEN,
        pad_idx,
    ).to(device)

    vae = VAETransformerDecoder(
        encoder=enc,
        vocab_size=vocab_size,
        pad_token=pad_idx,
        bos_token=bos_idx,
    ).to(device)

    checkpoint = torch.load(cfg.model_path, map_location=device)

    if "bundle_version" in checkpoint:
        sur = Z2MemorySurrogate(
            d_model=EMB_DIM,
            latent_dim=LATENT_DIM,
            max_len=MAX_LEN,
            layers=2,
            heads=4,
            ffn_dim=3 * EMB_DIM,
            dropout=DROPOUT,
        ).to(device)
        model = VAEWithSurrogate(vae, sur).to(device)
        if "vae" in checkpoint:
            load_res = model.vae.load_state_dict(checkpoint["vae"], strict=False)
            if load_res.missing_keys:
                logger.warning(
                    "Missing keys in VAE state dict: %s", load_res.missing_keys
                )
            else:
                logger.info("No missing keys in VAE state dict")

            if load_res.unexpected_keys:
                logger.warning(
                    "Unexpected keys in VAE state dict: %s", load_res.unexpected_keys
                )
            else:
                logger.info("No unexpected keys in VAE state dict")

        if "surrogate" in checkpoint:
            sur_res = model.surrogate.load_state_dict(
                checkpoint["surrogate"], strict=False
            )
            if sur_res.missing_keys:
                logger.warning(
                    "Missing keys in surrogate state dict: %s", sur_res.missing_keys
                )
            else:
                logger.info("No missing keys in surrogate state dict")

            if sur_res.unexpected_keys:
                logger.warning(
                    "Unexpected keys in surrogate state dict: %s",
                    sur_res.unexpected_keys,
                )
            else:
                logger.info("No unexpected keys in surrogate state dict")
        logger.info("Loaded VAE with surrogate from %s on %s", cfg.model_path, device)
    else:
        model = VAEWithSurrogate(vae, None).to(device)
        load_res = model.vae.load_state_dict(
            checkpoint.get("model_sd", checkpoint), strict=False
        )
        if load_res.missing_keys:
            logger.warning("Missing keys in VAE state dict: %s", load_res.missing_keys)
        else:
            logger.info("No missing keys in VAE state dict")

        if load_res.unexpected_keys:
            logger.warning(
                "Unexpected keys in VAE state dict: %s", load_res.unexpected_keys
            )
        else:
            logger.info("No unexpected keys in VAE state dict")

        logger.info("Loaded VAE from %s on %s", cfg.model_path, device)

    model.eval()
    return model
