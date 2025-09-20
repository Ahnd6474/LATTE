import pathlib
import sys

import pickle
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vae_module.config import Config
from vae_module.exceptions import CheckpointLoadError
from vae_module.loader import _load_checkpoint, load_vae
from vae_module.model import (
    SmallTransformer,
    VAETransformerDecoder,
    VAEWithSurrogate,
    Z2MemorySurrogate,
    EMB_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    FFN_DIM,
    MAX_LEN,
    LATENT_DIM,
    DROPOUT,
)


def test__load_checkpoint_retries_with_weights_only(monkeypatch):
    device = torch.device("cpu")
    calls = []

    def fake_load(path, *args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("weights_only") is False:
            return {"ok": True}
        raise pickle.UnpicklingError("weights-only mode incompatible")

    monkeypatch.setattr(torch, "load", fake_load)

    checkpoint = _load_checkpoint("dummy.pt", device)

    assert checkpoint == {"ok": True}
    assert calls == [
        {"map_location": device},
        {"map_location": device, "weights_only": False},
    ]


def test__load_checkpoint_invalid_file(tmp_path):
    device = torch.device("cpu")
    bad_path = tmp_path / "bad.pt"
    bad_path.write_bytes(b"\r\n")

    with pytest.raises(CheckpointLoadError) as excinfo:
        _load_checkpoint(str(bad_path), device)

    assert "git lfs pull" in str(excinfo.value).lower()


def test_load_vae_raises_for_invalid_checkpoint():
    cfg = Config(model_path=str(ROOT / "models" / "ep002.pt"), device="cpu")
    vocab_size = 32
    pad_idx = 0
    bos_idx = 1

    with pytest.raises(CheckpointLoadError):
        load_vae(cfg, vocab_size, pad_idx, bos_idx)


def test_load_vae_loads_diag_bias(tmp_path):
    vocab_size = 32
    pad_idx = 0
    bos_idx = 1

    encoder = SmallTransformer(
        vocab_size,
        EMB_DIM,
        NUM_LAYERS,
        NUM_HEADS,
        FFN_DIM,
        MAX_LEN,
        pad_idx,
    )
    vae = VAETransformerDecoder(
        encoder=encoder,
        vocab_size=vocab_size,
        pad_token=pad_idx,
        bos_token=bos_idx,
    )
    surrogate = Z2MemorySurrogate(
        d_model=EMB_DIM,
        latent_dim=LATENT_DIM,
        max_len=MAX_LEN,
        layers=2,
        heads=4,
        ffn_dim=3 * EMB_DIM,
        dropout=DROPOUT,
    )
    model = VAEWithSurrogate(vae, surrogate)

    diag_state = model.diag_bias.state_dict()
    diag_state["alpha"] = torch.tensor(0.25)
    diag_state["a_raw"] = torch.tensor([0.15])
    diag_state["d_raw"] = torch.tensor([-0.05])

    checkpoint = {
        "bundle_version": 3,
        "vae": model.vae.state_dict(),
        "surrogate": model.surrogate.state_dict(),
        "diag_bias": diag_state,
        "meta": {},
    }

    ckpt_path = tmp_path / "diag_bias.pt"
    torch.save(checkpoint, ckpt_path)

    cfg = Config(model_path=str(ckpt_path), device="cpu")
    loaded = load_vae(cfg, vocab_size, pad_idx, bos_idx)

    assert torch.allclose(loaded.diag_bias.alpha, diag_state["alpha"])
    assert torch.allclose(loaded.diag_bias.a_raw, diag_state["a_raw"])
    assert torch.allclose(loaded.diag_bias.d_raw, diag_state["d_raw"])
