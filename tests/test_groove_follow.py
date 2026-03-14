from __future__ import annotations

import numpy as np

from hardcoded_improv.chord_detector import ChordEvent
from hardcoded_improv.improv_engine import (
    GrooveConfig,
    HumanizeConfig,
    LickConfig,
    PhraseConfig,
    build_groove_grid_from_audio,
    generate_improv_events,
)


def _make_click_audio(sr: int = 22050, bpm: float = 110.0, duration_s: float = 6.0) -> np.ndarray:
    n = int(sr * duration_s)
    y = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    click_len = max(4, int(0.01 * sr))
    click = np.hanning(click_len).astype(np.float32)
    t = 0.0
    while t < duration_s:
        idx = int(t * sr)
        end = min(n, idx + click_len)
        seg = end - idx
        if seg > 0:
            y[idx:end] += click[:seg]
        t += period
    return np.clip(y, -1.0, 1.0)


def _chords() -> list[ChordEvent]:
    return [
        ChordEvent(0.0, 2.0, "E", "maj", 0.9),
        ChordEvent(2.0, 4.0, "C#", "min", 0.9),
        ChordEvent(4.0, 6.0, "A", "min", 0.9),
    ]


def test_groove_grid_non_empty_and_monotonic() -> None:
    audio = _make_click_audio()
    grid, scores, beats, bpm = build_groove_grid_from_audio(audio, 22050, swing_ratio=0.5)

    assert grid.size > 0
    assert scores.size == grid.size
    assert beats.size >= 2
    assert bpm > 0
    assert np.all(np.diff(grid) >= 0)


def test_events_align_to_grid_and_within_duration() -> None:
    sr = 22050
    bpm = 110.0
    bars = 4
    beats_per_bar = 4
    total_dur = bars * beats_per_bar * (60.0 / bpm)

    audio = _make_click_audio(sr=sr, bpm=bpm, duration_s=total_dur + 0.5)
    grid, scores, beats, _ = build_groove_grid_from_audio(audio, sr, swing_ratio=0.5)

    events = generate_improv_events(
        bpm=bpm,
        chord_timeline=_chords(),
        play_bars=bars,
        beats_per_bar=beats_per_bar,
        seed=42,
        humanize=HumanizeConfig(swing=0.0, jitter_ms=0.0, phrase_gap_prob=0.0),
        lick_cfg=LickConfig(lick_prob_on_boundary=0.0, lick_prob_on_phrase_start=0.0, grace_note_prob=0.0, slide_prob=0.0),
        phrase_cfg=PhraseConfig(phrase_len_bars=2, enable_call_response=False),
        groove_cfg=GrooveConfig(enabled=True, lock_strength=1.0, density_influence=0.7),
        groove_grid_times=grid,
        groove_onset_scores=scores,
        groove_beat_times=beats,
    )

    assert len(events) > 0
    g = np.asarray(grid, dtype=np.float64)
    for e in events:
        assert float(np.min(np.abs(g - e.time_sec))) <= 1e-4
        assert 0.0 <= e.time_sec <= total_dur + 0.2


def test_resolution_near_chord_boundaries() -> None:
    sr = 22050
    bpm = 110.0
    bars = 4
    total_dur = bars * 4 * (60.0 / bpm)
    audio = _make_click_audio(sr=sr, bpm=bpm, duration_s=total_dur + 0.5)
    grid, scores, beats, _ = build_groove_grid_from_audio(audio, sr, swing_ratio=0.5)

    events = generate_improv_events(
        bpm=bpm,
        chord_timeline=_chords(),
        play_bars=bars,
        beats_per_bar=4,
        seed=99,
        humanize=HumanizeConfig(swing=0.0, jitter_ms=0.0, phrase_gap_prob=0.0),
        lick_cfg=LickConfig(lick_prob_on_boundary=0.0, lick_prob_on_phrase_start=0.0, grace_note_prob=0.0, slide_prob=0.0),
        groove_cfg=GrooveConfig(enabled=True, lock_strength=1.0, density_influence=0.8),
        groove_grid_times=grid,
        groove_onset_scores=scores,
        groove_beat_times=beats,
    )

    boundaries = [2.0, 4.0]
    near = 0
    for b in boundaries:
        if any(abs(e.time_sec - b) <= 0.20 for e in events):
            near += 1
    assert near >= 1
