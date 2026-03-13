from __future__ import annotations

import numpy as np

from hardcoded_improv.tempo_estimator import (
    compute_listen_seconds,
    estimate_bar_length_seconds,
    estimate_beat_times,
    estimate_bpm,
)


def make_click_track(bpm: float, sr: int, duration_s: float) -> np.ndarray:
    n = int(sr * duration_s)
    y = np.zeros(n, dtype=np.float32)

    period_s = 60.0 / bpm
    click_len = int(0.01 * sr)
    window = np.hanning(max(click_len, 4)).astype(np.float32)

    t = 0.0
    while t < duration_s:
        idx = int(t * sr)
        end = min(n, idx + click_len)
        seg_len = end - idx
        if seg_len > 0:
            y[idx:end] += window[:seg_len]
        t += period_s

    y = np.clip(y, -1.0, 1.0)
    return y


def test_estimate_bpm_on_click_track_120() -> None:
    sr = 22050
    audio = make_click_track(bpm=120.0, sr=sr, duration_s=12.0)

    bpm = estimate_bpm(audio, sr)
    assert abs(bpm - 120.0) <= 3.0


def test_estimate_beat_times_on_click_track_120() -> None:
    sr = 22050
    target_bpm = 120.0
    audio = make_click_track(bpm=target_bpm, sr=sr, duration_s=12.0)

    beat_times = estimate_beat_times(audio, sr)
    assert beat_times.size >= 8

    diffs = np.diff(beat_times)
    median_period = float(np.median(diffs))
    expected_period = 60.0 / target_bpm
    assert abs(median_period - expected_period) <= 0.08


def test_estimate_bpm_falls_back_to_previous_for_silence() -> None:
    sr = 22050
    silence = np.zeros(sr * 4, dtype=np.float32)

    bpm = estimate_bpm(silence, sr, previous_bpm=110.0)
    assert bpm == 110.0


def test_bar_and_listen_helpers() -> None:
    bar_len = estimate_bar_length_seconds(120.0, beats_per_bar=4)
    assert abs(bar_len - 2.0) < 1e-6

    listen_s = compute_listen_seconds(120.0, bars=2, beats_per_bar=4)
    assert abs(listen_s - 4.0) < 1e-6
