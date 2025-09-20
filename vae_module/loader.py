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

def load_vae(cfg: Config, vocab_size: int, pad_idx: int, bos_idx: int) -> VAEWithSurrogate:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceNotAvailableError(cfg.device)

    enc = SmallTransformer(
        vocab_size, EMB_DIM, NUM_LAYERS, NUM_HEADS, FFN_DIM, MAX_LEN, pad_idx
    ).to(device)

    vae = VAETransformerDecoder(
        encoder=enc, vocab_size=vocab_size, pad_token=pad_idx, bos_token=bos_idx
    ).to(device)

    # ---- robust checkpoint load ----
    raw_ckpt = _load_checkpoint(cfg.model_path, device)
    # 언래핑
    if isinstance(raw_ckpt, Mapping):
        if "model_state" in raw_ckpt:
            ckpt = raw_ckpt["model_state"]
        elif "state_dict" in raw_ckpt:
            ckpt = raw_ckpt["state_dict"]
        else:
            ckpt = raw_ckpt
    else:
        ckpt = raw_ckpt

    # 모델 골격 구성 (surrogate는 bundle 여부와 무관하게 만들어 두어도 무방)
    sur = Z2MemorySurrogate(
        d_model=EMB_DIM, latent_dim=LATENT_DIM, max_len=MAX_LEN,
        layers=2, heads=4, ffn_dim=3 * EMB_DIM, dropout=DROPOUT
    ).to(device)
    model = VAEWithSurrogate(vae, sur).to(device)

    # 키 분석
    keys = list(ckpt.keys()) if isinstance(ckpt, Mapping) else []
    has_prefixed = any(k.startswith(("vae.", "surrogate.", "diag_bias.", "sur_ln", "sur_gate")) for k in keys)

    load_errors = []

    if has_prefixed:
        # 1) 전체 모델로 바로 로드 시도
        try:
            load_res = model.load_state_dict(ckpt, strict=False)
            if load_res.missing_keys:
                logger.warning("Missing keys (full model): %s", load_res.missing_keys)
            else:
                logger.info("No missing keys (full model)")
            if load_res.unexpected_keys:
                logger.warning("Unexpected keys (full model): %s", load_res.unexpected_keys)
            else:
                logger.info("No unexpected keys (full model)")
            logger.info("Loaded FULL model from %s on %s", cfg.model_path, device)
        except Exception as e:
            load_errors.append(e)
            # 2) 실패 시 prefix를 벗겨 부분 로드
            vae_sd = {k[len("vae."):] : v for k, v in ckpt.items() if k.startswith("vae.")}
            sur_sd = {k[len("surrogate."):] : v for k, v in ckpt.items() if k.startswith("surrogate.")}
            diag_sd = {k[len("diag_bias."):] : v for k, v in ckpt.items() if k.startswith("diag_bias.")}

            if vae_sd:
                res = model.vae.load_state_dict(vae_sd, strict=False)
                if res.missing_keys:
                    logger.warning("Missing VAE keys: %s", res.missing_keys)
                if res.unexpected_keys:
                    logger.warning("Unexpected VAE keys: %s", res.unexpected_keys)

            if sur_sd and model.surrogate is not None:
                res = model.surrogate.load_state_dict(sur_sd, strict=False)
                if res.missing_keys:
                    logger.warning("Missing surrogate keys: %s", res.missing_keys)
                if res.unexpected_keys:
                    logger.warning("Unexpected surrogate keys: %s", res.unexpected_keys)

            # diag_bias, sur_ln, sur_gate가 현재 클래스에 있으면 수동 주입 필요할 수 있음.
            # (state_dict로 바로 로드가 안 되면, 여기서 getattr/setattr로 파라미터 이름 맞춰 주입)
            logger.info("Loaded model by splitting prefixed keys from %s on %s", cfg.model_path, device)

    else:
        # 구(舊) 포맷: 순수 VAE 가중치만 있거나, 이름이 언프리픽스 상태
        try:
            res = model.vae.load_state_dict(ckpt.get("model_sd", ckpt), strict=False)
            if res.missing_keys:
                logger.warning("Missing keys in VAE state dict: %s", res.missing_keys)
            else:
                logger.info("No missing keys in VAE state dict")
            if res.unexpected_keys:
                logger.warning("Unexpected keys in VAE state dict: %s", res.unexpected_keys)
            else:
                logger.info("No unexpected keys in VAE state dict")
            logger.info("Loaded VAE from %s on %s", cfg.model_path, device)
        except Exception as e:
            load_errors.append(e)
            raise

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
            raise _checkpoint_error(path, inner_exc) from inner_exc
    except (RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise _checkpoint_error(path, exc) from exc
