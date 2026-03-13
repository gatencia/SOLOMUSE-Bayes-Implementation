from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

from hardcoded_improv.audio_io import LiveAudioInput, list_input_devices
from hardcoded_improv.config import AppConfig, load_config
from hardcoded_improv.utils import save_wav_mono, setup_logging

logger = logging.getLogger(__name__)

PENTATONIC_MIDI = [60, 62, 64, 67, 69]


def _run_phase(engine: LiveAudioInput, phase_name: str, duration_s: float, improv: bool) -> None:
    t0 = time.monotonic()
    next_note_t = t0

    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration_s:
            break

        level_db = engine.buffered_level_dbfs(0.25)
        logger.info("[%s] level=%6.2f dBFS", phase_name, level_db)

        if improv and time.monotonic() >= next_note_t:
            midi_note = random.choice(PENTATONIC_MIDI)
            logger.info("[play] improvised note=%s", midi_note)
            next_note_t = time.monotonic() + 0.4

        time.sleep(0.2)


def run_demo(cfg: AppConfig) -> None:
    logger.info("Starting live improv demo")
    with LiveAudioInput(cfg) as engine:
        if cfg.listen_seconds > 0:
            logger.info("Entering listen phase (%.2fs)", cfg.listen_seconds)
            _run_phase(engine, "listen", cfg.listen_seconds, improv=False)

        if cfg.play_seconds > 0:
            logger.info("Entering play phase (%.2fs)", cfg.play_seconds)
            _run_phase(engine, "play", cfg.play_seconds, improv=True)

        debug_audio = engine.get_last_seconds(10.0)
        save_wav_mono(cfg.debug_wav_path, debug_audio, cfg.sample_rate)
        logger.info("Saved debug audio: %s", cfg.debug_wav_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardcoded live improv demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--list-devices", action="store_true", help="List available input devices")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.list_devices:
        for idx, name in list_input_devices():
            print(f"{idx}: {name}")
        return

    cfg = load_config(Path(args.config))
    setup_logging(cfg.log_level)

    try:
        run_demo(cfg)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
