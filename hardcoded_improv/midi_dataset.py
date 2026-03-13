from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import mido

from hardcoded_improv.scale_utils import pentatonic_notes

logger = logging.getLogger(__name__)

DEFAULT_BPM = 120.0
DEFAULT_TEMPO_US_PER_BEAT = mido.bpm2tempo(DEFAULT_BPM)


@dataclass
class MidiNote:
    pitch: int
    start_tick: int
    end_tick: int


def _iter_midi_files(midi_dir: str | Path) -> list[Path]:
    root = Path(midi_dir)
    files = sorted([*root.rglob("*.mid"), *root.rglob("*.midi")])
    return files


def _choose_melody_track(mid: mido.MidiFile) -> mido.MidiTrack:
    best_track = mid.tracks[0]
    best_count = -1
    for tr in mid.tracks:
        count = 0
        for msg in tr:
            if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                count += 1
        if count > best_count:
            best_count = count
            best_track = tr
    return best_track


def _extract_notes_from_track(track: mido.MidiTrack) -> list[MidiNote]:
    abs_tick = 0
    active: dict[int, list[int]] = {}
    out: list[MidiNote] = []

    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, []).append(abs_tick)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            starts = active.get(msg.note)
            if starts:
                st = starts.pop(0)
                if abs_tick > st:
                    out.append(MidiNote(pitch=int(msg.note), start_tick=int(st), end_tick=int(abs_tick)))

    out.sort(key=lambda n: n.start_tick)
    return out


def _first_tempo(mid: mido.MidiFile) -> int:
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                return int(msg.tempo)
    return int(DEFAULT_TEMPO_US_PER_BEAT)


def _quantized_beat_pos(note_start_tick: int, ticks_per_beat: int) -> int:
    beats = note_start_tick / float(ticks_per_beat)
    beat_in_bar = beats % 4.0
    eighth = int(round(beat_in_bar * 2.0)) % 8
    return eighth


def _pitch_to_degree_and_octave(pitch: int, root_pc: int, quality: str) -> tuple[int, int]:
    scale = pentatonic_notes(root_pc, mode="major" if quality == "maj" else "minor")
    pc = pitch % 12

    best_degree = 0
    best_dist = 99
    for i, s in enumerate(scale):
        d = min((pc - s) % 12, (s - pc) % 12)
        if d < best_dist:
            best_dist = d
            best_degree = i

    octave_offset = int(round((pitch - 72) / 12.0))
    octave_offset = max(-2, min(2, octave_offset))
    return best_degree, octave_offset


def extract_training_samples_from_midi(
    midi_path: str | Path,
    default_chord_root: int = 0,
    default_chord_quality: str = "maj",
) -> list[dict[str, int | str]]:
    mid = mido.MidiFile(str(midi_path))
    melody_track = _choose_melody_track(mid)
    notes = _extract_notes_from_track(melody_track)
    if len(notes) < 2:
        return []

    tempo = _first_tempo(mid)
    bpm = mido.tempo2bpm(tempo)
    if not (30.0 <= bpm <= 300.0):
        bpm = DEFAULT_BPM

    logger.debug("MIDI %s tempo=%.2f notes=%d", midi_path, bpm, len(notes))

    samples: list[dict[str, int | str]] = []
    prev_degree, prev_oct = _pitch_to_degree_and_octave(notes[0].pitch, default_chord_root, default_chord_quality)
    prev_pitch = notes[0].pitch

    for n in notes[1:]:
        next_degree, next_oct = _pitch_to_degree_and_octave(n.pitch, default_chord_root, default_chord_quality)
        beat_pos = _quantized_beat_pos(n.start_tick, mid.ticks_per_beat)
        direction = 0
        if n.pitch > prev_pitch:
            direction = 1
        elif n.pitch < prev_pitch:
            direction = -1

        samples.append(
            {
                "prev_degree": int(prev_degree),
                "beat_pos": int(beat_pos),
                "chord_root": int(default_chord_root),
                "chord_quality": str(default_chord_quality),
                "prev_interval_direction": int(direction),
                "prev_octave_offset": int(prev_oct),
                "next_degree": int(next_degree),
                "next_octave_offset": int(next_oct),
            }
        )
        prev_degree, prev_oct = next_degree, next_oct
        prev_pitch = n.pitch

    return samples


def build_training_dataset(
    midi_dir: str | Path,
    default_chord_root: int = 0,
    default_chord_quality: str = "maj",
) -> list[dict[str, int | str]]:
    files = _iter_midi_files(midi_dir)
    dataset: list[dict[str, int | str]] = []
    for path in files:
        try:
            dataset.extend(
                extract_training_samples_from_midi(
                    path,
                    default_chord_root=default_chord_root,
                    default_chord_quality=default_chord_quality,
                )
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)

    logger.info("Built dataset: files=%d samples=%d", len(files), len(dataset))
    return dataset
