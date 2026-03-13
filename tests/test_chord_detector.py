from __future__ import annotations

import numpy as np

from hardcoded_improv.chord_detector import (
    ChordEvent,
    compute_chroma,
    detect_chord_from_chroma,
    detect_chords_over_time,
    infer_key_from_chords,
)


NOTE_TO_FREQ = {
    "C": 261.63,
    "C#": 277.18,
    "D": 293.66,
    "D#": 311.13,
    "E": 329.63,
    "F": 349.23,
    "F#": 369.99,
    "G": 392.00,
    "G#": 415.30,
    "A": 440.00,
    "A#": 466.16,
    "B": 493.88,
}


def _make_chord_tone(root: str, quality: str, sr: int, duration_s: float) -> np.ndarray:
    intervals = [0, 4, 7] if quality == "maj" else [0, 3, 7]
    semitone = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    root_pc = semitone[root]
    freqs = []
    for i in intervals:
        pc = (root_pc + i) % 12
        name = [k for k, v in semitone.items() if v == pc][0]
        freqs.append(NOTE_TO_FREQ[name])

    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    y = np.zeros(n, dtype=np.float32)
    for f in freqs:
        y += (0.28 * np.sin(2.0 * np.pi * f * t)).astype(np.float32)
        y += (0.08 * np.sin(2.0 * np.pi * 2.0 * f * t)).astype(np.float32)

    fade_len = max(16, int(0.02 * sr))
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    y[:fade_len] *= fade
    y[-fade_len:] *= fade[::-1]

    return np.clip(y, -1.0, 1.0)


def test_detect_chord_from_handcrafted_chroma() -> None:
    cmaj = np.zeros(12, dtype=np.float32)
    cmaj[[0, 4, 7]] = [1.0, 0.9, 0.8]
    root, quality, conf = detect_chord_from_chroma(cmaj)
    assert root == "C"
    assert quality == "maj"
    assert conf > 0.2

    amin = np.zeros(12, dtype=np.float32)
    amin[[9, 0, 4]] = [1.0, 0.9, 0.8]
    root, quality, conf = detect_chord_from_chroma(amin)
    assert root == "A"
    assert quality == "min"
    assert conf > 0.2


def test_infer_key_from_chords_progression() -> None:
    seq = [
        ChordEvent(0.0, 1.0, "C", "maj", 0.9),
        ChordEvent(1.0, 2.0, "G", "maj", 0.8),
        ChordEvent(2.0, 3.0, "A", "min", 0.8),
        ChordEvent(3.0, 4.0, "F", "maj", 0.8),
    ]
    key = infer_key_from_chords(seq)
    assert key == "C major"


def test_detect_chords_over_time_synthetic_audio() -> None:
    sr = 22050
    cmaj = _make_chord_tone("C", "maj", sr=sr, duration_s=2.0)
    gmaj = _make_chord_tone("G", "maj", sr=sr, duration_s=2.0)
    audio = np.concatenate([cmaj, gmaj]).astype(np.float32)

    chroma = compute_chroma(audio[:sr], sr)
    assert chroma.shape == (12,)

    beats = np.arange(0.0, 4.01, 0.5, dtype=np.float32)
    events = detect_chords_over_time(audio, sr, frame_sec=0.5, beat_times=beats)

    assert len(events) >= 2
    assert events[0].root_note_name == "C"
    assert events[0].quality == "maj"

    roots = [e.root_note_name for e in events]
    assert "G" in roots

    for ev in events:
        nearest_start = beats[np.argmin(np.abs(beats - ev.start_sec))]
        assert abs(float(nearest_start) - ev.start_sec) <= 1e-5
