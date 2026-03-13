from __future__ import annotations

import random
import importlib

from hardcoded_improv.bayes_model import BayesianNoteModel, NoteContext
from hardcoded_improv.midi_dataset import build_training_dataset


def _write_synthetic_midi(path: str) -> None:
    mido = importlib.import_module("mido")
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set explicit tempo.
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

    # Monophonic lead line in C major pentatonic.
    notes = [60, 62, 64, 67, 69, 67, 64, 62, 60]
    for i, n in enumerate(notes):
        track.append(mido.Message("note_on", note=n, velocity=90, time=0 if i == 0 else 0))
        track.append(mido.Message("note_off", note=n, velocity=0, time=240))  # eighth note

    mid.save(path)


def test_train_and_sample_on_synthetic_midi(tmp_path) -> None:
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_path = midi_dir / "lead.mid"
    _write_synthetic_midi(str(midi_path))

    samples = build_training_dataset(midi_dir)
    assert len(samples) >= 6

    model = BayesianNoteModel(laplace=1.0)
    model.fit(samples)

    ctx = NoteContext(
        prev_degree=0,
        beat_pos=0,
        chord_quality="maj",
        chord_root=0,
        prev_interval_direction=1,
        prev_octave_offset=0,
    )
    degree, octv = model.sample_next(ctx, random.Random(123))

    assert 0 <= degree <= 4
    assert -2 <= octv <= 2

    out = tmp_path / "model.json"
    model.save_json(str(out))
    loaded = BayesianNoteModel.load_json(str(out))

    degree2, octv2 = loaded.sample_next(ctx, random.Random(123))
    assert 0 <= degree2 <= 4
    assert -2 <= octv2 <= 2
