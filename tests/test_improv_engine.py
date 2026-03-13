from __future__ import annotations

from hardcoded_improv.chord_detector import ChordEvent
from hardcoded_improv.improv_engine import generate_improv_events, loop_chord_progression
from hardcoded_improv.midi_out import build_scheduled_events


def _base_chords() -> list[ChordEvent]:
    return [
        ChordEvent(0.0, 2.0, "C", "maj", 0.8),
        ChordEvent(2.0, 4.0, "G", "maj", 0.8),
    ]


def test_generate_events_is_sorted_and_deterministic() -> None:
    events_a = generate_improv_events(120.0, _base_chords(), play_bars=4, seed=42)
    events_b = generate_improv_events(120.0, _base_chords(), play_bars=4, seed=42)

    assert len(events_a) > 0
    assert [e.time_sec for e in events_a] == sorted(e.time_sec for e in events_a)

    sig_a = [(e.time_sec, e.midi_note, e.velocity, e.duration_sec) for e in events_a]
    sig_b = [(e.time_sec, e.midi_note, e.velocity, e.duration_sec) for e in events_b]
    assert sig_a == sig_b


def test_loop_chord_progression_fills_duration() -> None:
    out = loop_chord_progression(_base_chords(), total_duration_sec=10.0)
    assert len(out) >= 4
    assert out[0].start_sec == 0.0
    assert out[-1].end_sec <= 10.0


def test_scheduled_event_order() -> None:
    events = generate_improv_events(120.0, _base_chords(), play_bars=2, seed=7)
    sched = build_scheduled_events(events)

    assert len(sched) >= 2
    for i in range(1, len(sched)):
        prev = sched[i - 1]
        cur = sched[i]
        assert (prev.time_sec, 0 if prev.kind == "off" else 1) <= (
            cur.time_sec,
            0 if cur.kind == "off" else 1,
        )
