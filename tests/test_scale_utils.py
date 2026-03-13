from __future__ import annotations

from hardcoded_improv.scale_utils import pentatonic_notes


def test_major_pentatonic_pitch_classes() -> None:
    assert pentatonic_notes("C", mode="major") == [0, 2, 4, 7, 9]
    assert pentatonic_notes("D", mode="major") == [2, 4, 6, 9, 11]


def test_minor_pentatonic_pitch_classes() -> None:
    assert pentatonic_notes("A", mode="minor") == [9, 0, 2, 4, 7]
    assert pentatonic_notes(0, mode="minor") == [0, 3, 5, 7, 10]
