from __future__ import annotations

NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def note_name_to_pc(note_name: str) -> int:
    if note_name not in NOTE_TO_PC:
        raise ValueError(f"Unknown note name: {note_name}")
    return NOTE_TO_PC[note_name]


def pentatonic_notes(root: str | int, mode: str = "major") -> list[int]:
    """Return pentatonic scale pitch classes relative to root.

    Returns 5 pitch classes in range [0, 11].
    """
    if isinstance(root, str):
        root_pc = note_name_to_pc(root)
    else:
        root_pc = int(root) % 12

    mode_l = mode.lower()
    if mode_l not in {"major", "minor"}:
        raise ValueError("mode must be 'major' or 'minor'")

    if mode_l == "major":
        intervals = [0, 2, 4, 7, 9]
    else:
        intervals = [0, 3, 5, 7, 10]

    return [int((root_pc + i) % 12) for i in intervals]
