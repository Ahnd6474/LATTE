import pickle
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch

from .config import Config
from .exceptions import CheckpointLoadError, DeviceNotAvailableError
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

    checkpoint = _load_checkpoint(cfg.model_path, device)

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
        vae_state = checkpoint.get("model_sd", checkpoint)
        load_res = model.vae.load_state_dict(vae_state, strict=False)
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


def _checkpoint_error(path: str, original: Exception) -> CheckpointLoadError:
    """Return a :class:`CheckpointLoadError` enriched with troubleshooting hints."""

    hints = []
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
        if size == 0:
            hints.append("The file is empty.")
        elif size < 1024:
            hints.append(
                "The file is only "
                f"{size} bytes and looks like a placeholder rather than a checkpoint. "
                "Run `git lfs pull` to download the actual weights or provide a full"
                " checkpoint file."
            )
        with file_path.open("rb") as handle:
            head = handle.read(256)
        if b"git-lfs" in head:
            hints.append(
                "It appears to be a Git LFS pointer. Run `git lfs pull` to download the"
                " actual weights."
            )
    except OSError:
        # If the file cannot be inspected we keep the original error message.
        pass

    if hints:
        message = f"{original}. {' '.join(hints)}"
        wrapped = RuntimeError(message)
        return CheckpointLoadError(path, wrapped)
    return CheckpointLoadError(path, original)


def _load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a checkpoint while remaining compatible with legacy PyTorch saves."""

    load_kwargs = {"map_location": device}
    try:
        return torch.load(path, **load_kwargs)
    except pickle.UnpicklingError as exc:
        logger.info(
            "torch.load failed due to weights-only mode; retrying with weights_only=False"
        )
        try:
            return torch.load(path, weights_only=False, **load_kwargs)
        except Exception as inner_exc:  # pragma: no cover - fallback should rarely fail
            raise _checkpoint_error(path, inner_exc) from inner_exc
    except (RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise _checkpoint_error(path, exc) from exc
