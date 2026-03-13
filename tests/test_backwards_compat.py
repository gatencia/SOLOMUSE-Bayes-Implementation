from __future__ import annotations

from hardcoded_improv.chord_detector import ChordEvent
from hardcoded_improv.improv_engine import generate_improv_events


def test_generate_improv_events_old_signature_still_works() -> None:
    chords = [
        ChordEvent(0.0, 2.0, "C", "maj", 0.9),
        ChordEvent(2.0, 4.0, "G", "maj", 0.9),
    ]

    # Old call shape (positional through bayes_model) should still work.
    events = generate_improv_events(120.0, chords, 4, 4, 42, 5, None)

    assert len(events) > 0
    assert all(e.duration_sec > 0 for e in events)
