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

    with pytest.raises(CheckpointLoadError):
        _load_checkpoint(str(bad_path), device)


def test_load_vae_raises_for_invalid_checkpoint():
    cfg = Config(model_path=str(ROOT / "models" / "ep002.pt"), device="cpu")
    vocab_size = 32
    pad_idx = 0
    bos_idx = 1

    with pytest.raises(CheckpointLoadError):
        load_vae(cfg, vocab_size, pad_idx, bos_idx)
