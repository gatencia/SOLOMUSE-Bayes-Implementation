from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    sample_rate: int = 44100
    blocksize: int = 512
    input_device: int | str | None = None
    channels: int = 1
    ring_buffer_seconds: int = 20
    listen_seconds: float = 4.0
    play_seconds: float = 8.0
    log_level: str = "INFO"
    debug_wav_path: str = "debug_last10s.wav"


def _validate_config(cfg: AppConfig) -> AppConfig:
    if cfg.sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if cfg.blocksize <= 0:
        raise ValueError("blocksize must be > 0")
    if cfg.channels != 1:
        raise ValueError("Only mono input is supported for now (channels must be 1)")
    if cfg.ring_buffer_seconds < 10:
        raise ValueError("ring_buffer_seconds must be >= 10")
    if cfg.listen_seconds < 0:
        raise ValueError("listen_seconds must be >= 0")
    if cfg.play_seconds < 0:
        raise ValueError("play_seconds must be >= 0")
    return cfg


def load_config(config_path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    if config_path is None:
        return _validate_config(cfg)

    path = Path(config_path)
    if not path.exists():
        return _validate_config(cfg)

    with path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = yaml.safe_load(f) or {}

    for key, value in payload.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    return _validate_config(cfg)
