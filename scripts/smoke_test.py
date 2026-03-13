from __future__ import annotations

from pathlib import Path

from hardcoded_improv.cli import run_demo
from hardcoded_improv.config import load_config
from hardcoded_improv.utils import setup_logging


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "config.yaml")

    cfg.listen_seconds = 4.0
    cfg.play_seconds = 6.0
    cfg.debug_wav_path = str(project_root / "smoke_last10s.wav")

    setup_logging(cfg.log_level)
    run_demo(cfg)


if __name__ == "__main__":
    main()
