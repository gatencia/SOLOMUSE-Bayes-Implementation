from __future__ import annotations

import logging

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover - optional fallback path
    librosa = None

try:
    import aubio
except Exception:  # pragma: no cover - optional fallback path
    aubio = None

logger = logging.getLogger(__name__)

MIN_BPM = 40.0
MAX_BPM = 240.0
DEFAULT_BPM_FALLBACK = 120.0
DEFAULT_HOP_LENGTH = 512
DEFAULT_WIN_LENGTH = 2048

_last_valid_bpm = DEFAULT_BPM_FALLBACK
MIN_ONSET_STD = 1e-3
MIN_ONSET_MEAN = 1e-4


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


def _estimate_with_aubio(audio: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    if aubio is None:
        return float("nan"), np.array([], dtype=np.float32)

    hop_s = DEFAULT_HOP_LENGTH
    win_s = DEFAULT_WIN_LENGTH
    tempo_obj = aubio.tempo("default", win_s, hop_s, sr)

    y = _to_mono_float32(audio)
    beat_times: list[float] = []
    for i in range(0, y.size, hop_s):
        frame = y[i : i + hop_s]
        if frame.size < hop_s:
            frame = np.pad(frame, (0, hop_s - frame.size))
        is_beat = tempo_obj(frame.astype(np.float32))
        if is_beat:
            beat_times.append(float(tempo_obj.get_last_s()))

    beats = np.asarray(beat_times, dtype=np.float32)
    if beats.size >= 2:
        median_period = float(np.median(np.diff(beats)))
        bpm = 60.0 / median_period if median_period > 0 else float("nan")
    else:
        bpm = float(tempo_obj.get_bpm())

    return bpm, beats


def estimate_bpm(audio: np.ndarray, sr: int, previous_bpm: float | None = None) -> float:
    """Estimate BPM from audio using onset strength and tempo estimation."""
    global _last_valid_bpm

    y = _to_mono_float32(audio)
    if y.size < max(sr, DEFAULT_WIN_LENGTH):
        fallback = previous_bpm if previous_bpm is not None else _last_valid_bpm
        logger.warning("Audio too short for tempo estimation. Using fallback BPM=%.2f", fallback)
        return float(fallback)

    if librosa is None:
        aubio_bpm, _ = _estimate_with_aubio(y, sr)
        if _valid_bpm(aubio_bpm):
            _last_valid_bpm = float(aubio_bpm)
            logger.info("Tempo metrics: backend=aubio chosen_bpm=%.2f", aubio_bpm)
            return float(aubio_bpm)

        fallback = previous_bpm if previous_bpm is not None else _last_valid_bpm
        logger.warning("No tempo backend available. Using fallback BPM=%.2f", fallback)
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

    onset_reliable = onset_mean >= MIN_ONSET_MEAN and onset_std >= MIN_ONSET_STD

    if not onset_reliable:
        if previous_bpm is not None and _valid_bpm(previous_bpm):
            chosen_bpm = float(previous_bpm)
            reason = "previous_low_onset"
        elif _valid_bpm(_last_valid_bpm):
            chosen_bpm = float(_last_valid_bpm)
            reason = "last_valid_low_onset"
        else:
            chosen_bpm = DEFAULT_BPM_FALLBACK
            reason = "default_low_onset"

        logger.info(
            "Tempo metrics: onset_mean=%.6f onset_std=%.6f raw_bpm=nan chosen_bpm=%.2f source=%s",
            onset_mean,
            onset_std,
            chosen_bpm,
            reason,
        )
        return float(chosen_bpm)

    raw_tempo = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        aggregate=np.median,
    )
    raw_bpm = float(raw_tempo[0]) if np.size(raw_tempo) else float("nan")

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        backtrack=False,
        units="frames",
    )
    onset_anchor_bpm = float("nan")
    if np.size(onset_frames) >= 2:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
        ioi = np.diff(onset_times)
        if ioi.size:
            median_ioi = float(np.median(ioi))
            if median_ioi > 0:
                onset_anchor_bpm = 60.0 / median_ioi

    candidate_bpm = raw_bpm
    if np.isfinite(raw_bpm) and np.isfinite(onset_anchor_bpm):
        candidates = [raw_bpm, raw_bpm / 2.0, raw_bpm * 2.0]
        valid = [c for c in candidates if _valid_bpm(c)]
        if valid:
            candidate_bpm = min(valid, key=lambda c: abs(c - onset_anchor_bpm))

    if _valid_bpm(candidate_bpm):
        chosen_bpm = float(candidate_bpm)
        _last_valid_bpm = chosen_bpm
        reason = "raw_or_octave_corrected"
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

    if librosa is None:
        _, beats = _estimate_with_aubio(y, sr)
        if beats.size:
            return beats
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

    if np.size(beat_frames) == 0:
        beat_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=DEFAULT_HOP_LENGTH,
            backtrack=False,
            units="frames",
        )

    if np.size(beat_frames) == 0 and bpm > 0:
        duration_s = y.size / float(sr)
        period_s = 60.0 / bpm
        grid = np.arange(0.0, duration_s, period_s, dtype=np.float32)
        logger.info("Beat detection fallback: generated grid beats count=%d", grid.size)
        return grid

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