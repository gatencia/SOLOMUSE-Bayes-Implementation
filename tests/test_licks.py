from __future__ import annotations

from hardcoded_improv.chord_detector import ChordEvent
from hardcoded_improv.improv_engine import LICK_TEMPLATES, HumanizeConfig, LickConfig, generate_improv_events


def _chords() -> list[ChordEvent]:
    return [
        ChordEvent(0.0, 2.0, "C", "maj", 0.9),
        ChordEvent(2.0, 4.0, "G", "maj", 0.9),
        ChordEvent(4.0, 6.0, "A", "min", 0.9),
        ChordEvent(6.0, 8.0, "F", "maj", 0.9),
    ]


def test_lick_templates_degree_bounds() -> None:
    for mode, templates in LICK_TEMPLATES.items():
        assert mode in {"maj", "min"}
        for tpl in templates:
            assert 3 <= len(tpl) <= 6
            assert all(0 <= int(d) <= 4 for d in tpl)


def test_lick_injection_deterministic() -> None:
    hz = HumanizeConfig(jitter_ms=0.0)
    lk = LickConfig(
        lick_prob_on_boundary=0.95,
        lick_prob_on_phrase_start=0.95,
        grace_note_prob=0.0,
        slide_prob=0.0,
        max_lick_len_steps=6,
    )
    a = generate_improv_events(120.0, _chords(), play_bars=8, seed=101, humanize_config=hz, lick_config=lk)
    b = generate_improv_events(120.0, _chords(), play_bars=8, seed=101, humanize_config=hz, lick_config=lk)

    sig_a = [(e.time_sec, e.midi_note, e.velocity, e.duration_sec) for e in a]
    sig_b = [(e.time_sec, e.midi_note, e.velocity, e.duration_sec) for e in b]
    assert sig_a == sig_b


def test_grace_notes_non_negative_and_valid_duration() -> None:
    hz = HumanizeConfig(jitter_ms=0.0, phrase_gap_prob=0.0)
    lk = LickConfig(
        lick_prob_on_boundary=0.0,
        lick_prob_on_phrase_start=0.0,
        grace_note_prob=0.95,
        slide_prob=0.0,
    )
    events = generate_improv_events(120.0, _chords(), play_bars=6, seed=5, humanize_config=hz, lick_config=lk)

    assert len(events) > 0
    assert all(e.time_sec >= 0.0 for e in events)
    assert all(e.duration_sec > 0.0 for e in events)
    assert all(36 <= e.midi_note <= 96 for e in events)


def test_event_count_increases_with_high_lick_and_grace_probs() -> None:
    hz = HumanizeConfig(jitter_ms=0.0, phrase_gap_prob=0.0)
    low = LickConfig(
        lick_prob_on_boundary=0.0,
        lick_prob_on_phrase_start=0.0,
        grace_note_prob=0.0,
        slide_prob=0.0,
    )
    high = LickConfig(
        lick_prob_on_boundary=0.95,
        lick_prob_on_phrase_start=0.95,
        grace_note_prob=0.95,
        slide_prob=0.2,
    )

    base_events = generate_improv_events(120.0, _chords(), play_bars=8, seed=11, humanize_config=hz, lick_config=low)
    rich_events = generate_improv_events(120.0, _chords(), play_bars=8, seed=11, humanize_config=hz, lick_config=high)

    assert len(rich_events) > len(base_events)
