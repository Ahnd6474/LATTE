import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """Configuration object for VAE settings."""

    model_path: str
    device: str = "cpu"
    batch_size: int = 64
    max_len: int = 512

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(**data)


def load_config(path: str) -> Config:
    """Load a configuration from YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix in {".yml", ".yaml"}:
        data = yaml.safe_load(p.read_text())
    else:
        data = json.loads(p.read_text())
    return Config.from_dict(data)
