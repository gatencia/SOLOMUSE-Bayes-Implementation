from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
import logging
from pathlib import Path
import time
import wave

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

from hardcoded_improv.audio_io import LiveAudioInput, list_input_devices
from hardcoded_improv.bayes_model import BayesianNoteModel
from hardcoded_improv.chord_detector import ChordEvent, detect_chords_over_time
from hardcoded_improv.config import AppConfig
from hardcoded_improv.improv_engine import (
    GrooveConfig,
    HumanizeConfig,
    LickConfig,
    NoteEvent,
    PhraseConfig,
    build_groove_grid_from_audio,
    generate_improv_events,
    loop_chord_progression,
)
from hardcoded_improv.midi_out import MidiOut, play_events_realtime
from hardcoded_improv.tempo_estimator import (
    compute_listen_seconds,
    estimate_bar_length_seconds,
    estimate_beat_times,
    estimate_bpm,
)
from hardcoded_improv.utils import save_wav_mono

logger = logging.getLogger(__name__)


@dataclass
class LiveDemoResult:
    bpm: float
    chord_count: int
    event_count: int
    listen_audio_path: str
    chords_json_path: str
    events_csv_path: str
    output_mid_path: str | None


def _select_input_device(preferred_name: str = "scarlett") -> int:
    candidates = list_input_devices()
    if not candidates:
        raise RuntimeError("No input devices available")

    preferred_l = preferred_name.lower()
    for idx, name in candidates:
        if preferred_l in name.lower():
            logger.info("Selected input device: %d (%s)", idx, name)
            return idx

    known = ", ".join([f"{idx}:{name}" for idx, name in candidates[:10]])
    raise RuntimeError(f"Preferred input device containing '{preferred_name}' not found. Available: {known}")


def _ensure_artifacts_dir(artifacts_dir: str | Path) -> Path:
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_wav_mono_float32(path: str | Path) -> tuple[np.ndarray, int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input WAV not found: {p}")

    with wave.open(str(p), "rb") as wf:
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        width = int(wf.getsampwidth())
        frames = int(wf.getnframes())
        raw = wf.readframes(frames)

    if width == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        # support PCM32 via int32 interpretation
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {width} bytes")

    if channels > 1:
        x = x.reshape(-1, channels)[:, 0]
    return x.reshape(-1).astype(np.float32), sr


def _save_chords_json(path: Path, chords: list[ChordEvent], bpm: float) -> None:
    payload = {
        "bpm": float(bpm),
        "chords": [asdict(c) for c in chords],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_events_csv(path: Path, events: list[NoteEvent]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "midi_note", "velocity", "duration_sec"])
        for ev in events:
            w.writerow([f"{ev.time_sec:.6f}", ev.midi_note, ev.velocity, f"{ev.duration_sec:.6f}"])


def _save_output_mid(path: Path, events: list[NoteEvent], bpm: float) -> None:
    try:
        import mido
    except Exception as exc:
        raise RuntimeError("Cannot write MIDI file because mido is unavailable") from exc

    ticks_per_beat = 480
    tempo = mido.bpm2tempo(float(bpm))
    ticks_per_second = ticks_per_beat * (float(bpm) / 60.0)

    timeline: list[tuple[float, str, int, int]] = []
    for ev in events:
        timeline.append((ev.time_sec, "note_on", ev.midi_note, ev.velocity))
        timeline.append((ev.time_sec + ev.duration_sec, "note_off", ev.midi_note, 0))
    timeline.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    tr.append(mido.MetaMessage("track_name", name="SOLOMUSE Lead", time=0))
    tr.append(mido.Message("program_change", program=29, channel=0, time=0))

    last_tick = 0
    for t_sec, kind, note, vel in timeline:
        tick = int(round(t_sec * ticks_per_second))
        delta = max(0, tick - last_tick)
        tr.append(mido.Message(kind, note=int(note), velocity=int(vel), time=delta))
        last_tick = tick

    mid.save(str(path))


def _extract_groove_template(
    audio: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    beats_per_bar: int = 4,
) -> tuple[list[float], list[float]]:
    slot_count = beats_per_bar * 2
    if librosa is None or beat_times.size < 2:
        return [0.0] * slot_count, [0.5] * slot_count

    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        units="time",
        backtrack=False,
    )
    onset_times = np.asarray(onset_frames, dtype=np.float32)

    offs: list[list[float]] = [[] for _ in range(slot_count)]
    hits = np.zeros(slot_count, dtype=np.float64)
    occ = np.zeros(slot_count, dtype=np.float64)

    for b in range(len(beat_times) - 1):
        t0 = float(beat_times[b])
        t1 = float(beat_times[b + 1])
        beat_dur = max(1e-4, t1 - t0)
        beat_in_bar = b % beats_per_bar
        for sub in range(2):
            slot = beat_in_bar * 2 + sub
            expected = t0 + (sub * 0.5) * beat_dur
            occ[slot] += 1.0

            if onset_times.size == 0:
                continue

            idx = int(np.argmin(np.abs(onset_times - expected)))
            dt = float(onset_times[idx] - expected)
            if abs(dt) <= min(0.09, beat_dur * 0.35):
                offs[slot].append(dt)
                hits[slot] += 1.0

    offsets: list[float] = []
    density: list[float] = []
    for s in range(slot_count):
        if offs[s]:
            offsets.append(float(np.median(np.asarray(offs[s], dtype=np.float64))))
        else:
            offsets.append(0.0)

        d = hits[s] / occ[s] if occ[s] > 0 else 0.0
        density.append(float(max(0.0, min(1.0, d))))

    return offsets, density


def _compute_pipeline(
    listen_audio: np.ndarray,
    sr: int,
    listen_bars: int,
    play_bars: int,
    seed: int | None,
    bayes_model_path: str | None,
    groove_lock: float = 0.98,
    groove_gate: float = 0.28,
    ornament_level: float = 0.35,
) -> tuple[float, list[ChordEvent], list[NoteEvent]]:
    bpm = estimate_bpm(listen_audio, sr, previous_bpm=120.0)
    if not np.isfinite(bpm) or bpm <= 0:
        raise RuntimeError("Failed to estimate BPM from listen audio")

    grid_times, grid_scores, beat_times, _ = build_groove_grid_from_audio(listen_audio, sr, swing_ratio=0.5)
    groove_offsets, groove_density = _extract_groove_template(listen_audio, sr, beat_times, beats_per_bar=4)
    chords = detect_chords_over_time(listen_audio, sr, frame_sec=0.5, beat_times=beat_times)
    if not chords:
        raise RuntimeError("Chord timeline is empty; cannot continue demo")

    bar_len = estimate_bar_length_seconds(bpm, beats_per_bar=4)
    play_duration = bar_len * play_bars
    play_chords = loop_chord_progression(chords, total_duration_sec=play_duration)

    bayes_model = None
    if bayes_model_path:
        model_path = Path(bayes_model_path)
        if not model_path.exists():
            raise RuntimeError(f"Bayes model not found: {model_path}")
        bayes_model = BayesianNoteModel.load_json(str(model_path))

    orn = float(max(0.0, min(1.0, ornament_level)))

    events = generate_improv_events(
        bpm=bpm,
        chord_timeline=play_chords,
        play_bars=play_bars,
        beats_per_bar=4,
        seed=seed,
        bayes_model=bayes_model,
        humanize=HumanizeConfig(
            swing=0.0,
            jitter_ms=4.0,
            vel_jitter=4,
            phrase_len_bars=2,
            phrase_gap_prob=0.05,
            staccato_prob=0.15,
            legato_prob=0.25,
            min_dur_frac=0.45,
            max_dur_frac=0.9,
        ),
        lick_cfg=LickConfig(
            lick_prob_on_boundary=0.08 * orn,
            lick_prob_on_phrase_start=0.12 * orn,
            grace_note_prob=0.05 * orn,
            slide_prob=0.03 * orn,
            max_lick_len_steps=4,
        ),
        phrase_cfg=PhraseConfig(
            phrase_len_bars=2,
            enable_call_response=True,
            strong_target_prob=0.82,
            approach_prob=0.78,
            max_jump_semitones=7,
        ),
        groove_cfg=GrooveConfig(
            enabled=True,
            lock_strength=max(0.0, min(1.0, groove_lock)),
            density_influence=0.75,
            max_offset_sec=0.06,
            base_play_prob=0.04,
            density_play_scale=0.84,
            strong_beat_bonus=0.25,
            min_density_gate=max(0.0, min(0.8, groove_gate)),
        ),
        groove_offsets=groove_offsets,
        groove_density=groove_density,
        groove_grid_times=grid_times,
        groove_onset_scores=grid_scores,
        groove_beat_times=beat_times,
    )
    if not events:
        raise RuntimeError("Generated event list is empty")

    return float(bpm), chords, events


def run_live_demo(
    cfg: AppConfig,
    midi_port: str | None,
    listen_bars: int = 2,
    play_bars: int = 16,
    artifacts_dir: str | Path = "artifacts/live_demo",
    seed: int | None = None,
    bayes_model_path: str | None = None,
    output_mid: bool = False,
    dry_run: bool = False,
    prefer_input_name: str = "scarlett",
    groove_lock: float = 0.98,
    groove_gate: float = 0.28,
    ornament_level: float = 0.35,
) -> LiveDemoResult:
    if listen_bars <= 0 or play_bars <= 0:
        raise ValueError("listen_bars and play_bars must be > 0")

    out_dir = _ensure_artifacts_dir(artifacts_dir)

    cfg_use = cfg
    if cfg.input_device is None:
        cfg_use = replace(cfg, input_device=_select_input_device(prefer_input_name))

    # Two-bar listen duration based on a fast bootstrap tempo estimate.
    with LiveAudioInput(cfg_use) as engine:
        bootstrap_sec = 4.0
        time.sleep(bootstrap_sec)
        bootstrap_audio = engine.get_last_seconds(bootstrap_sec)
        bootstrap_bpm = estimate_bpm(bootstrap_audio, cfg_use.sample_rate, previous_bpm=120.0)
        listen_seconds = compute_listen_seconds(bootstrap_bpm, bars=listen_bars, beats_per_bar=4)
        logger.info("Listen phase: bpm=%.2f listen_seconds=%.2f", bootstrap_bpm, listen_seconds)

        if listen_seconds > bootstrap_sec:
            time.sleep(listen_seconds - bootstrap_sec)
        listen_audio = engine.get_last_seconds(listen_seconds)

    bpm, chords, events = _compute_pipeline(
        listen_audio,
        cfg_use.sample_rate,
        listen_bars=listen_bars,
        play_bars=play_bars,
        seed=seed,
        bayes_model_path=bayes_model_path,
        groove_lock=groove_lock,
        groove_gate=groove_gate,
        ornament_level=ornament_level,
    )

    listen_wav = out_dir / "listen_audio.wav"
    chords_json = out_dir / "chords.json"
    events_csv = out_dir / "events.csv"
    out_mid = out_dir / "output.mid"

    save_wav_mono(listen_wav, listen_audio, cfg_use.sample_rate)
    _save_chords_json(chords_json, chords, bpm)
    _save_events_csv(events_csv, events)

    if output_mid:
        _save_output_mid(out_mid, events, bpm)

    logger.info("Detected BPM: %.2f", bpm)
    for c in chords:
        logger.info("Chord: %.2f -> %.2f  %s:%s (%.2f)", c.start_sec, c.end_sec, c.root_note_name, c.quality, c.confidence)

    if dry_run:
        play_events_realtime(events, dry_run=True)
    else:
        with MidiOut(port_name=midi_port) as out:
            play_events_realtime(events, midi_out=out, dry_run=False)

    return LiveDemoResult(
        bpm=bpm,
        chord_count=len(chords),
        event_count=len(events),
        listen_audio_path=str(listen_wav),
        chords_json_path=str(chords_json),
        events_csv_path=str(events_csv),
        output_mid_path=str(out_mid) if output_mid else None,
    )


def run_simulation_demo(
    cfg: AppConfig,
    input_wav: str | Path,
    listen_bars: int = 2,
    play_bars: int = 16,
    artifacts_dir: str | Path = "artifacts/live_demo_sim",
    seed: int | None = None,
    bayes_model_path: str | None = None,
    output_mid: bool = True,
    full_wav_listen: bool = False,
    groove_lock: float = 0.98,
    groove_gate: float = 0.28,
    ornament_level: float = 0.35,
) -> LiveDemoResult:
    if listen_bars <= 0 or play_bars <= 0:
        raise ValueError("listen_bars and play_bars must be > 0")

    out_dir = _ensure_artifacts_dir(artifacts_dir)
    audio, sr = _load_wav_mono_float32(input_wav)
    if sr != cfg.sample_rate:
        raise RuntimeError(
            f"Input WAV sample rate ({sr}) does not match config sample_rate ({cfg.sample_rate})"
        )

    bootstrap_sec = 4.0
    if audio.size < int(bootstrap_sec * sr):
        raise RuntimeError("Input WAV is too short for bootstrap tempo estimation")
    bootstrap_audio = audio[: int(bootstrap_sec * sr)]
    bootstrap_bpm = estimate_bpm(bootstrap_audio, sr, previous_bpm=120.0)

    if full_wav_listen:
        listen_audio = audio
        logger.info(
            "Simulation listen mode: full WAV (%.2fs)",
            audio.size / float(sr),
        )
    else:
        listen_seconds = compute_listen_seconds(bootstrap_bpm, bars=listen_bars, beats_per_bar=4)
        listen_samples = int(round(listen_seconds * sr))
        if audio.size < listen_samples:
            raise RuntimeError(
                f"Input WAV is too short for listen phase: need {listen_seconds:.2f}s, got {audio.size / sr:.2f}s"
            )
        listen_audio = audio[:listen_samples]
    bpm, chords, events = _compute_pipeline(
        listen_audio,
        sr,
        listen_bars=listen_bars,
        play_bars=play_bars,
        seed=seed,
        bayes_model_path=bayes_model_path,
        groove_lock=groove_lock,
        groove_gate=groove_gate,
        ornament_level=ornament_level,
    )

    listen_wav = out_dir / "listen_audio.wav"
    chords_json = out_dir / "chords.json"
    events_csv = out_dir / "events.csv"
    out_mid = out_dir / "output.mid"

    save_wav_mono(listen_wav, listen_audio, sr)
    _save_chords_json(chords_json, chords, bpm)
    _save_events_csv(events_csv, events)
    if output_mid:
        _save_output_mid(out_mid, events, bpm)

    logger.info(
        "Simulation complete: bpm=%.2f chords=%d events=%d", bpm, len(chords), len(events)
    )
    return LiveDemoResult(
        bpm=bpm,
        chord_count=len(chords),
        event_count=len(events),
        listen_audio_path=str(listen_wav),
        chords_json_path=str(chords_json),
        events_csv_path=str(events_csv),
        output_mid_path=str(out_mid) if output_mid else None,
    )
