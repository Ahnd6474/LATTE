import torch

from .config import Config
from .exceptions import DeviceNotAvailableError
from .logger import setup_logger
from .model import SmallTransformer, VAETransformerDecoder, EMB_DIM, NUM_LAYERS, NUM_HEADS, FFN_DIM, MAX_LEN

logger = setup_logger(__name__)


def load_vae(cfg: Config, vocab_size: int, pad_idx: int, bos_idx: int) -> VAETransformerDecoder:
    """Load VAE model from checkpoint defined in Config."""
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

    model = VAETransformerDecoder(
        encoder=enc,
        vocab_size=vocab_size,
        pad_token=pad_idx,
        bos_token=bos_idx,
    ).to(device)

    checkpoint = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_sd"])
    model.eval()
    logger.info("Loaded VAE from %s on %s", cfg.model_path, device)
    return model
