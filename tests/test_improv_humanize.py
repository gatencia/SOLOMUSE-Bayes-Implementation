from __future__ import annotations

import numpy as np

from hardcoded_improv.chord_detector import ChordEvent
from hardcoded_improv.improv_engine import HumanizeConfig, generate_improv_events


def _chords() -> list[ChordEvent]:
    return [
        ChordEvent(0.0, 4.0, "C", "maj", 0.9),
        ChordEvent(4.0, 8.0, "G", "maj", 0.9),
        ChordEvent(8.0, 12.0, "A", "min", 0.9),
        ChordEvent(12.0, 16.0, "F", "maj", 0.9),
    ]


def test_humanize_deterministic_with_seed() -> None:
    hz = HumanizeConfig(swing=0.18, jitter_ms=8.0, phrase_gap_prob=0.2)
    a = generate_improv_events(120.0, _chords(), play_bars=8, seed=123, humanize_config=hz)
    b = generate_improv_events(120.0, _chords(), play_bars=8, seed=123, humanize_config=hz)

    sig_a = [(e.time_sec, e.midi_note, e.velocity, e.duration_sec) for e in a]
    sig_b = [(e.time_sec, e.midi_note, e.velocity, e.duration_sec) for e in b]
    assert sig_a == sig_b


def test_humanize_times_durations_velocities_valid() -> None:
    hz = HumanizeConfig(swing=0.2, jitter_ms=10.0)
    evs = generate_improv_events(120.0, _chords(), play_bars=8, seed=7, humanize_config=hz)
    assert len(evs) > 0

    times = [e.time_sec for e in evs]
    assert all(t >= 0.0 for t in times)
    assert times == sorted(times)

    assert all(e.duration_sec > 0.0 for e in evs)
    assert all(1 <= e.velocity <= 127 for e in evs)


def test_humanize_swing_delays_offbeats_more() -> None:
    bpm = 120.0
    step = 60.0 / bpm / 2.0
    hz = HumanizeConfig(
        swing=0.25,
        jitter_ms=0.0,
        vel_jitter=0,
        phrase_gap_prob=0.0,
        staccato_prob=0.0,
        legato_prob=0.0,
    )

    evs = generate_improv_events(bpm, _chords(), play_bars=8, seed=99, humanize_config=hz)

    on_delays = []
    off_delays = []
    for e in evs:
        k = int(round(e.time_sec / step))
        d = float(e.time_sec - k * step)
        if k % 2 == 0:
            on_delays.append(d)
        else:
            off_delays.append(d)

    assert len(on_delays) > 5
    assert len(off_delays) > 5
    assert float(np.mean(off_delays)) > float(np.mean(on_delays)) + 0.02
