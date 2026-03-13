from __future__ import annotations

from dataclasses import dataclass
import random

from hardcoded_improv.chord_detector import ChordEvent
from hardcoded_improv.scale_utils import note_name_to_pc, pentatonic_notes


@dataclass
class NoteEvent:
    time_sec: float
    midi_note: int
    velocity: int
    duration_sec: float


def _mode_from_quality(quality: str) -> str:
    return "major" if quality == "maj" else "minor"


def _pc_to_midi(pc: int, octave: int = 5) -> int:
    return int(12 * (octave + 1) + (pc % 12))


def _chord_at_time(chords: list[ChordEvent], t: float) -> ChordEvent:
    if not chords:
        return ChordEvent(0.0, 1e9, "C", "maj", 0.0)
    for ch in chords:
        if ch.start_sec <= t < ch.end_sec:
            return ch
    return chords[-1]


def _motif_rng_pattern(rng: random.Random) -> list[int]:
    length = rng.randint(2, 4)
    return [rng.randint(0, 4) for _ in range(length)]


def generate_improv_events(
    bpm: float,
    chord_timeline: list[ChordEvent],
    play_bars: int = 8,
    beats_per_bar: int = 4,
    seed: int | None = None,
    base_octave: int = 5,
) -> list[NoteEvent]:
    """Generate deterministic rule-based improv MIDI events.

    Rhythm: mostly eighth notes with occasional rests.
    Musical rule: favor chord root near chord boundaries.
    """
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    if play_bars <= 0:
        return []

    rng = random.Random(seed)
    sec_per_beat = 60.0 / bpm
    step = sec_per_beat / 2.0  # eighth notes
    total_beats = play_bars * beats_per_bar
    total_steps = total_beats * 2
    note_dur = step * 0.85

    motif = _motif_rng_pattern(rng)
    motif_pos = 0
    events: list[NoteEvent] = []

    boundary_times = {round(ch.start_sec, 3) for ch in chord_timeline}

    for i in range(total_steps):
        t = i * step
        chord = _chord_at_time(chord_timeline, t)
        mode = _mode_from_quality(chord.quality)
        scale_pcs = pentatonic_notes(chord.root_note_name, mode=mode)
        root_pc = note_name_to_pc(chord.root_note_name)

        # Occasional rest (about 20%).
        if rng.random() < 0.2:
            continue

        near_boundary = round(t, 3) in boundary_times
        if near_boundary and rng.random() < 0.75:
            chosen_pc = root_pc
            velocity = rng.randint(88, 112)
        else:
            idx = motif[motif_pos % len(motif)]
            motif_pos += 1
            chosen_pc = scale_pcs[idx % len(scale_pcs)]
            velocity = rng.randint(70, 105)

        # Small octave movement.
        octave = base_octave + rng.choice([-1, 0, 0, 0, 1])
        midi_note = _pc_to_midi(chosen_pc, octave=octave)
        midi_note = max(36, min(96, midi_note))

        events.append(
            NoteEvent(
                time_sec=float(t),
                midi_note=int(midi_note),
                velocity=int(velocity),
                duration_sec=float(note_dur),
            )
        )

    events.sort(key=lambda e: e.time_sec)
    return events


def loop_chord_progression(chords: list[ChordEvent], total_duration_sec: float) -> list[ChordEvent]:
    if total_duration_sec <= 0:
        return []
    if not chords:
        return [ChordEvent(0.0, total_duration_sec, "C", "maj", 0.0)]

    listened_end = max(ch.end_sec for ch in chords)
    if listened_end <= 0:
        return [ChordEvent(0.0, total_duration_sec, chords[0].root_note_name, chords[0].quality, chords[0].confidence)]

    out: list[ChordEvent] = []
    k = 0
    while True:
        shift = k * listened_end
        for ch in chords:
            s = ch.start_sec + shift
            e = ch.end_sec + shift
            if s >= total_duration_sec:
                return out
            out.append(
                ChordEvent(
                    start_sec=s,
                    end_sec=min(e, total_duration_sec),
                    root_note_name=ch.root_note_name,
                    quality=ch.quality,
                    confidence=ch.confidence,
                )
            )
        k += 1

