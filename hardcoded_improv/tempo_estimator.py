from __future__ import annotations

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

MIN_BPM = 40.0
MAX_BPM = 240.0
DEFAULT_BPM_FALLBACK = 120.0
DEFAULT_HOP_LENGTH = 512
DEFAULT_WIN_LENGTH = 2048

_last_valid_bpm = DEFAULT_BPM_FALLBACK


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y[:, 0]
        else:
            y = np.mean(y, axis=1)
    return y.reshape(-1)


def _valid_bpm(bpm: float) -> bool:
    return np.isfinite(bpm) and MIN_BPM <= bpm <= MAX_BPM


def estimate_bpm(audio: np.ndarray, sr: int, previous_bpm: float | None = None) -> float:
    """Estimate BPM from audio using onset strength and tempo estimation."""
    global _last_valid_bpm

    y = _to_mono_float32(audio)
    if y.size < max(sr, DEFAULT_WIN_LENGTH):
        fallback = previous_bpm if previous_bpm is not None else _last_valid_bpm
        logger.warning("Audio too short for tempo estimation. Using fallback BPM=%.2f", fallback)
        return float(fallback)

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        n_fft=DEFAULT_WIN_LENGTH,
        aggregate=np.median,
    )

    onset_mean = float(np.mean(onset_env)) if onset_env.size else 0.0
    onset_std = float(np.std(onset_env)) if onset_env.size else 0.0

    raw_tempo = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        aggregate=np.median,
    )
    raw_bpm = float(raw_tempo[0]) if np.size(raw_tempo) else float("nan")

    if _valid_bpm(raw_bpm):
        chosen_bpm = raw_bpm
        _last_valid_bpm = raw_bpm
        reason = "raw"
    else:
        if previous_bpm is not None and _valid_bpm(previous_bpm):
            chosen_bpm = float(previous_bpm)
            reason = "previous"
        elif _valid_bpm(_last_valid_bpm):
            chosen_bpm = float(_last_valid_bpm)
            reason = "last_valid"
        else:
            chosen_bpm = DEFAULT_BPM_FALLBACK
            reason = "default"

    logger.info(
        "Tempo metrics: onset_mean=%.6f onset_std=%.6f raw_bpm=%.2f chosen_bpm=%.2f source=%s",
        onset_mean,
        onset_std,
        raw_bpm,
        chosen_bpm,
        reason,
    )
    return float(chosen_bpm)


def estimate_beat_times(audio: np.ndarray, sr: int) -> np.ndarray:
    """Estimate beat times (seconds) from audio."""
    y = _to_mono_float32(audio)
    if y.size < max(sr, DEFAULT_WIN_LENGTH):
        return np.array([], dtype=np.float32)

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        n_fft=DEFAULT_WIN_LENGTH,
        aggregate=np.median,
    )
    bpm = estimate_bpm(y, sr)

    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        start_bpm=bpm,
        trim=False,
    )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
    return np.asarray(beat_times, dtype=np.float32)


def estimate_bar_length_seconds(bpm: float, beats_per_bar: int = 4) -> float:
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be > 0")
    return (60.0 / bpm) * beats_per_bar


def compute_listen_seconds(bpm: float, bars: int = 2, beats_per_bar: int = 4) -> float:
    if bars <= 0:
        raise ValueError("bars must be > 0")
    return estimate_bar_length_seconds(bpm, beats_per_bar=beats_per_bar) * bars