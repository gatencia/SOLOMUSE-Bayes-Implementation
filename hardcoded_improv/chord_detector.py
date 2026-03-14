from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover - optional runtime dependency path
    librosa = None

logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["maj", "min"]

DEFAULT_HOP_LENGTH = 512
MIN_FRAME_SEC = 0.25
TRANSITION_PENALTY = 0.08


@dataclass
class ChordEvent:
    start_sec: float
    end_sec: float
    root_note_name: str
    quality: str
    confidence: float

    @property
    def label(self) -> str:
        return f"{self.root_note_name}:{self.quality}"


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y[:, 0]
        else:
            y = np.mean(y, axis=1)
    return y.reshape(-1)


def _normalize_chroma(chroma: np.ndarray) -> np.ndarray:
    c = np.asarray(chroma, dtype=np.float32).reshape(12)
    c = np.maximum(c, 0.0)
    norm = np.linalg.norm(c)
    if norm <= 1e-9:
        return np.zeros(12, dtype=np.float32)
    return c / norm


def _build_templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    major = np.zeros(12, dtype=np.float32)
    major[[0, 4, 7]] = np.array([1.0, 0.9, 0.8], dtype=np.float32)

    minor = np.zeros(12, dtype=np.float32)
    minor[[0, 3, 7]] = np.array([1.0, 0.9, 0.8], dtype=np.float32)

    mats: list[np.ndarray] = []
    labels: list[tuple[int, str]] = []
    for root in range(12):
        mats.append(np.roll(major, root))
        labels.append((root, "maj"))
    for root in range(12):
        mats.append(np.roll(minor, root))
        labels.append((root, "min"))

    templates = np.vstack([_normalize_chroma(m) for m in mats])
    return templates, labels


_TEMPLATES, _TEMPLATE_LABELS = _build_templates()


def _chroma_matrix(audio: np.ndarray, sr: int) -> np.ndarray:
    y = _to_mono_float32(audio)
    if librosa is None or y.size == 0:
        return np.zeros((12, 0), dtype=np.float32)

    min_len = 4096
    if y.size < min_len:
        y = np.pad(y, (0, min_len - y.size))

    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=DEFAULT_HOP_LENGTH)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=DEFAULT_HOP_LENGTH, n_fft=2048)

    return np.asarray(chroma, dtype=np.float32)


def _chord_scores(chroma: np.ndarray) -> np.ndarray:
    c = _normalize_chroma(chroma)
    if np.allclose(c, 0.0):
        return np.zeros(len(_TEMPLATE_LABELS), dtype=np.float32)
    return _TEMPLATES @ c


def _decode_chord_scores(scores: np.ndarray) -> tuple[str, str, float]:
    if scores.size == 0:
        return "N", "none", 0.0

    best_idx = int(np.argmax(scores))
    sorted_scores = np.sort(scores)
    best = float(sorted_scores[-1])
    second = float(sorted_scores[-2]) if scores.size > 1 else -1.0

    root, quality = _TEMPLATE_LABELS[best_idx]
    confidence = float(np.clip((best - second) / 0.5, 0.0, 1.0))
    return NOTE_NAMES[root], quality, confidence


def compute_chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute a single normalized 12-bin chroma vector from an audio window."""
    chroma = _chroma_matrix(audio, sr)
    if chroma.size == 0:
        return np.zeros(12, dtype=np.float32)
    pooled = np.median(chroma, axis=1)
    return _normalize_chroma(pooled)


def detect_chord_from_chroma(chroma: np.ndarray) -> tuple[str, str, float]:
    """Detect root and quality from a 12-bin chroma vector using triad templates."""
    scores = _chord_scores(chroma)
    root, quality, confidence = _decode_chord_scores(scores)
    logger.debug("Chord detect: root=%s quality=%s conf=%.3f", root, quality, confidence)
    return root, quality, confidence


def _smooth_states(score_matrix: np.ndarray, transition_penalty: float = TRANSITION_PENALTY) -> np.ndarray:
    if score_matrix.ndim != 2 or score_matrix.shape[0] == 0:
        return np.array([], dtype=np.int32)

    n_frames, n_states = score_matrix.shape
    dp = np.full((n_frames, n_states), -1e9, dtype=np.float32)
    back = np.zeros((n_frames, n_states), dtype=np.int32)

    dp[0] = score_matrix[0]
    for t in range(1, n_frames):
        prev = dp[t - 1]
        for s in range(n_states):
            candidates = prev - transition_penalty
            candidates[s] = prev[s]
            j = int(np.argmax(candidates))
            dp[t, s] = score_matrix[t, s] + candidates[j]
            back[t, s] = j

    states = np.zeros(n_frames, dtype=np.int32)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]

    return states


def _snap_to_nearest_beat(time_s: float, beat_times: np.ndarray) -> float:
    if beat_times.size == 0:
        return float(time_s)
    idx = int(np.argmin(np.abs(beat_times - time_s)))
    return float(beat_times[idx])


def _merge_adjacent_same(events: list[ChordEvent]) -> list[ChordEvent]:
    if not events:
        return []
    out: list[ChordEvent] = [events[0]]
    for ev in events[1:]:
        last = out[-1]
        if ev.root_note_name == last.root_note_name and ev.quality == last.quality:
            dur_last = max(1e-6, last.end_sec - last.start_sec)
            dur_cur = max(1e-6, ev.end_sec - ev.start_sec)
            conf = (last.confidence * dur_last + ev.confidence * dur_cur) / (dur_last + dur_cur)
            out[-1] = ChordEvent(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, ev.end_sec),
                root_note_name=last.root_note_name,
                quality=last.quality,
                confidence=float(conf),
            )
        else:
            out.append(ev)
    return out


def _collapse_short_segments(events: list[ChordEvent], min_duration: float) -> list[ChordEvent]:
    if not events:
        return []
    if min_duration <= 0:
        return events

    work = list(events)
    i = 0
    while i < len(work):
        ev = work[i]
        dur = ev.end_sec - ev.start_sec
        if dur >= min_duration or len(work) == 1:
            i += 1
            continue

        # Merge short segment with neighboring segment preferring same label or higher confidence.
        left = work[i - 1] if i > 0 else None
        right = work[i + 1] if i + 1 < len(work) else None

        if left and right and left.root_note_name == right.root_note_name and left.quality == right.quality:
            # bridge into one
            merged_conf = float((left.confidence + right.confidence + ev.confidence) / 3.0)
            work[i - 1] = ChordEvent(
                start_sec=left.start_sec,
                end_sec=right.end_sec,
                root_note_name=left.root_note_name,
                quality=left.quality,
                confidence=merged_conf,
            )
            del work[i : i + 2]
            i = max(0, i - 1)
            continue

        if right is not None and (left is None or right.confidence >= left.confidence):
            work[i + 1] = ChordEvent(
                start_sec=ev.start_sec,
                end_sec=right.end_sec,
                root_note_name=right.root_note_name,
                quality=right.quality,
                confidence=float((right.confidence + ev.confidence) / 2.0),
            )
            del work[i]
            continue

        if left is not None:
            work[i - 1] = ChordEvent(
                start_sec=left.start_sec,
                end_sec=ev.end_sec,
                root_note_name=left.root_note_name,
                quality=left.quality,
                confidence=float((left.confidence + ev.confidence) / 2.0),
            )
            del work[i]
            i = max(0, i - 1)
            continue

        i += 1

    return work


def detect_chords_over_time(
    audio: np.ndarray,
    sr: int,
    frame_sec: float = 0.5,
    beat_times: np.ndarray | None = None,
    min_chord_sec: float | None = None,
) -> list[ChordEvent]:
    """Detect smoothed chord events over time and optionally snap boundaries to beats."""
    y = _to_mono_float32(audio)
    if y.size == 0:
        return []

    frame_sec = max(float(frame_sec), MIN_FRAME_SEC)
    frame_len = max(1, int(round(frame_sec * sr)))
    n_frames = int(np.ceil(y.size / frame_len))

    frame_scores: list[np.ndarray] = []
    frame_bounds: list[tuple[float, float]] = []
    for i in range(n_frames):
        start = i * frame_len
        end = min((i + 1) * frame_len, y.size)
        chunk = y[start:end]
        chroma = compute_chroma(chunk, sr)
        frame_scores.append(_chord_scores(chroma))
        frame_bounds.append((start / sr, end / sr))

    score_matrix = np.vstack(frame_scores) if frame_scores else np.zeros((0, 24), dtype=np.float32)
    states = _smooth_states(score_matrix)

    bt = np.asarray(beat_times if beat_times is not None else [], dtype=np.float32)
    events: list[ChordEvent] = []
    if states.size == 0:
        return events

    seg_start = 0
    for i in range(1, states.size + 1):
        boundary = i == states.size or states[i] != states[seg_start]
        if not boundary:
            continue

        s0 = frame_bounds[seg_start][0]
        s1 = frame_bounds[i - 1][1]
        s0 = _snap_to_nearest_beat(s0, bt)
        s1 = _snap_to_nearest_beat(s1, bt)
        if s1 <= s0:
            s1 = frame_bounds[i - 1][1]

        chord_idx = int(states[seg_start])
        root_idx, quality = _TEMPLATE_LABELS[chord_idx]

        conf = float(np.mean(np.clip(score_matrix[seg_start:i, chord_idx], 0.0, 1.0)))
        events.append(
            ChordEvent(
                start_sec=float(s0),
                end_sec=float(s1),
                root_note_name=NOTE_NAMES[root_idx],
                quality=quality,
                confidence=conf,
            )
        )
        seg_start = i

    events = _merge_adjacent_same(events)

    if min_chord_sec is None:
        if bt.size >= 2:
            beat_period = float(np.median(np.diff(bt)))
            min_chord_sec = max(frame_sec * 1.5, beat_period * 0.75)
        else:
            min_chord_sec = frame_sec * 1.5

    events = _collapse_short_segments(events, min_duration=float(min_chord_sec))
    events = _merge_adjacent_same(events)
    return events


def infer_key_from_chords(chord_sequence: list[ChordEvent]) -> str:
    """Infer key name from detected chords using a simple diatonic scoring heuristic."""
    if not chord_sequence:
        return "Unknown"

    major_pattern = {
        0: "maj",  # I
        2: "min",  # ii
        4: "min",  # iii
        5: "maj",  # IV
        7: "maj",  # V
        9: "min",  # vi
    }
    minor_pattern = {
        0: "min",  # i
        3: "maj",  # III
        5: "min",  # iv
        7: "min",  # v
        8: "maj",  # VI
        10: "maj",  # VII
    }

    best_key = "Unknown"
    best_score = -1.0

    for tonic in range(12):
        score_major = 0.0
        score_minor = 0.0
        for ev in chord_sequence:
            if ev.root_note_name not in NOTE_NAMES:
                continue
            chord_pc = NOTE_NAMES.index(ev.root_note_name)
            interval = (chord_pc - tonic) % 12
            w = max(ev.confidence, 0.2)
            if major_pattern.get(interval) == ev.quality:
                score_major += w
            if minor_pattern.get(interval) == ev.quality:
                score_minor += w

        if score_major > best_score:
            best_score = score_major
            best_key = f"{NOTE_NAMES[tonic]} major"
        if score_minor > best_score:
            best_score = score_minor
            best_key = f"{NOTE_NAMES[tonic]} minor"

    logger.info("Inferred key=%s score=%.3f", best_key, best_score)
    return best_key