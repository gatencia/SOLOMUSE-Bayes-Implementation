from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np

from hardcoded_improv.audio_io import LiveAudioInput, list_input_devices
from hardcoded_improv.chord_detector import detect_chords_over_time, infer_key_from_chords
from hardcoded_improv.config import AppConfig, load_config
from hardcoded_improv.improv_engine import generate_improv_events, loop_chord_progression
from hardcoded_improv.midi_out import MidiOut, play_events_realtime
from hardcoded_improv.tempo_estimator import compute_listen_seconds, estimate_bar_length_seconds, estimate_beat_times, estimate_bpm
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


def run_tempo_probe(cfg: AppConfig, seconds: float) -> None:
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    logger.info("Collecting %.2f seconds of audio for tempo estimation", seconds)
    with LiveAudioInput(cfg) as engine:
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= seconds:
                break
            level_db = engine.buffered_level_dbfs(0.25)
            logger.info("[tempo] level=%6.2f dBFS", level_db)
            time.sleep(0.2)

        audio = engine.get_last_seconds(seconds)

    bpm = estimate_bpm(audio, cfg.sample_rate)
    beat_times = estimate_beat_times(audio, cfg.sample_rate)

    print(f"Estimated BPM: {bpm:.2f}")
    if beat_times.size == 0:
        print("Beat times (s): []")
    else:
        rounded = np.round(beat_times, 3).tolist()
        print(f"Beat times (s): {rounded}")


def run_chords_probe(cfg: AppConfig, seconds: float) -> None:
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    logger.info("Collecting %.2f seconds of audio for chord detection", seconds)
    with LiveAudioInput(cfg) as engine:
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= seconds:
                break
            level_db = engine.buffered_level_dbfs(0.25)
            logger.info("[chords] level=%6.2f dBFS", level_db)
            time.sleep(0.2)

        audio = engine.get_last_seconds(seconds)

    bpm = estimate_bpm(audio, cfg.sample_rate)
    beat_times = estimate_beat_times(audio, cfg.sample_rate)
    events = detect_chords_over_time(audio, cfg.sample_rate, frame_sec=0.5, beat_times=beat_times)
    key_name = infer_key_from_chords(events)

    print(f"Estimated BPM: {bpm:.2f}")
    print(f"Inferred key: {key_name}")
    if not events:
        print("Chord timeline: []")
        return

    print("Chord timeline:")
    for ev in events:
        print(
            f"  {ev.start_sec:6.2f}s -> {ev.end_sec:6.2f}s  {ev.root_note_name}:{ev.quality}  conf={ev.confidence:.2f}"
        )


def run_improv_baseline(
    cfg: AppConfig,
    listen_bars: int,
    play_bars: int,
    seed: int | None,
    dry_run: bool,
    midi_port: str | None,
) -> None:
    if listen_bars <= 0:
        raise ValueError("listen_bars must be > 0")
    if play_bars <= 0:
        raise ValueError("play_bars must be > 0")

    logger.info("Starting baseline improv (listen_bars=%d, play_bars=%d)", listen_bars, play_bars)
    with LiveAudioInput(cfg) as engine:
        bootstrap_sec = 4.0
        logger.info("Bootstrap listening for %.2fs to estimate tempo", bootstrap_sec)
        time.sleep(bootstrap_sec)

        probe_audio = engine.get_last_seconds(bootstrap_sec)
        bpm = estimate_bpm(probe_audio, cfg.sample_rate, previous_bpm=120.0)
        listen_seconds = compute_listen_seconds(bpm, bars=listen_bars, beats_per_bar=4)

        logger.info("Estimated BPM=%.2f, listen_seconds=%.2f", bpm, listen_seconds)
        if listen_seconds > bootstrap_sec:
            time.sleep(listen_seconds - bootstrap_sec)

        listen_audio = engine.get_last_seconds(listen_seconds)

    beat_times = estimate_beat_times(listen_audio, cfg.sample_rate)
    listened_chords = detect_chords_over_time(listen_audio, cfg.sample_rate, frame_sec=0.5, beat_times=beat_times)
    if not listened_chords:
        logger.warning("No chords detected during listen phase. Falling back to C:maj")

    bar_len = estimate_bar_length_seconds(bpm, beats_per_bar=4)
    play_duration = bar_len * play_bars
    play_chords = loop_chord_progression(listened_chords, total_duration_sec=play_duration)

    events = generate_improv_events(
        bpm=bpm,
        chord_timeline=play_chords,
        play_bars=play_bars,
        beats_per_bar=4,
        seed=seed,
    )
    logger.info("Generated %d note events", len(events))

    if dry_run:
        play_events_realtime(events, dry_run=True)
        return

    with MidiOut(port_name=midi_port) as out:
        play_events_realtime(events, midi_out=out, dry_run=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardcoded live improv CLI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--list-devices", action="store_true", help="List available input devices")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Run listen/play improv demo")

    tempo_parser = subparsers.add_parser("tempo", help="Estimate tempo from live audio")
    tempo_parser.add_argument("--seconds", type=float, default=12.0, help="Rolling window size in seconds")

    chords_parser = subparsers.add_parser("chords", help="Estimate chord timeline from live audio")
    chords_parser.add_argument("--seconds", type=float, default=12.0, help="Rolling window size in seconds")

    improv_parser = subparsers.add_parser("improv", help="Run rule-based pentatonic improv over detected chords")
    improv_parser.add_argument("--listen-bars", type=int, default=2, help="Bars to listen before playing")
    improv_parser.add_argument("--play-bars", type=int, default=8, help="Bars to play")
    improv_parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed")
    improv_parser.add_argument("--dry-run", action="store_true", help="Print/schedule events without MIDI output")
    improv_parser.add_argument("--midi-port", type=str, default=None, help="MIDI output port name (optional)")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = load_config(Path(args.config))
    setup_logging(cfg.log_level)

    command = args.command or "run"

    if args.list_devices:
        for idx, name in list_input_devices():
            print(f"{idx}: {name}")
        return

    try:
        if command == "tempo":
            run_tempo_probe(cfg, seconds=args.seconds)
        elif command == "chords":
            run_chords_probe(cfg, seconds=args.seconds)
        elif command == "improv":
            run_improv_baseline(
                cfg,
                listen_bars=args.listen_bars,
                play_bars=args.play_bars,
                seed=args.seed,
                dry_run=args.dry_run,
                midi_port=args.midi_port,
            )
        else:
            run_demo(cfg)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
