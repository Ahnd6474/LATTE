import pickle
from collections import OrderedDict
from collections.abc import Mapping
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


def _strip_diag_bias_keys(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``state`` without entries belonging to ``CrossDiagBias``."""

    if not isinstance(state, Mapping):
        return state

    prefix = "diag_bias."
    has_diag_keys = any(
        isinstance(key, str) and key.startswith(prefix) for key in state.keys()
    )
    if not has_diag_keys:
        return state

    items = [
        (key, value)
        for key, value in state.items()
        if not (isinstance(key, str) and key.startswith(prefix))
    ]
    if isinstance(state, OrderedDict):
        return OrderedDict(items)
    return dict(items)


def _extract_diag_bias_state(
    checkpoint: Mapping[str, Any]
) -> Optional[OrderedDict[str, torch.Tensor]]:
    """Extract a ``CrossDiagBias`` state dict from a checkpoint if present."""

    if not isinstance(checkpoint, Mapping):
        return None

    diag_bias = checkpoint.get("diag_bias")
    if isinstance(diag_bias, Mapping):
        diag_items = OrderedDict(
            (str(key), value)
            for key, value in diag_bias.items()
            if isinstance(key, str)
        )
        if diag_items:
            return diag_items

    prefix = "diag_bias."

    def _from_mapping(state: Mapping[str, Any]) -> Optional[OrderedDict[str, torch.Tensor]]:
        if not isinstance(state, Mapping):
            return None
        extracted = OrderedDict(
            (key[len(prefix) :], value)
            for key, value in state.items()
            if isinstance(key, str) and key.startswith(prefix)
        )
        return extracted or None

    direct = _from_mapping(checkpoint)
    if direct is not None:
        return direct

    for parent_key in ("model_state", "model_sd", "sur_adapter", "vae", "surrogate"):
        state = checkpoint.get(parent_key)
        extracted = _from_mapping(state) if isinstance(state, Mapping) else None
        if extracted is not None:
            return extracted

    return None


def _load_diag_bias_state(model: VAEWithSurrogate, checkpoint: Mapping[str, Any]) -> None:
    """Load ``CrossDiagBias`` weights if the checkpoint provides them."""

    diag_bias = getattr(model, "diag_bias", None)
    if diag_bias is None:
        return

    diag_state = _extract_diag_bias_state(checkpoint)
    if diag_state is None:
        logger.info("No diag bias state found in checkpoint; using defaults")
        return

    load_res = diag_bias.load_state_dict(diag_state, strict=False)
    if load_res.missing_keys:
        logger.warning(
            "Missing keys in diag bias state dict: %s", load_res.missing_keys
        )
    else:
        logger.info("No missing keys in diag bias state dict")

    if load_res.unexpected_keys:
        logger.warning(
            "Unexpected keys in diag bias state dict: %s",
            load_res.unexpected_keys,
        )
    else:
        logger.info("No unexpected keys in diag bias state dict")


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
        _load_diag_bias_state(model, checkpoint)
        logger.info("Loaded VAE with surrogate from %s on %s", cfg.model_path, device)
    else:
        model = VAEWithSurrogate(vae, None).to(device)
        vae_state = checkpoint.get("model_sd", checkpoint)
        if isinstance(vae_state, Mapping):
            vae_state = _strip_diag_bias_keys(vae_state)
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

        _load_diag_bias_state(model, checkpoint)

        logger.info("Loaded VAE from %s on %s", cfg.model_path, device)

    model.eval()
    return model


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
            raise CheckpointLoadError(path, inner_exc) from inner_exc
    except (RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointLoadError(path, exc) from exc
