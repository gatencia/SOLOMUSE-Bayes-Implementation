from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave

import numpy as np

from hardcoded_improv.chord_detector import ChordEvent, detect_chords_over_time
from hardcoded_improv.improv_engine import generate_improv_events
from hardcoded_improv.tempo_estimator import estimate_beat_times, estimate_bpm


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        n = int(wf.getnframes())
        raw = wf.readframes(n)

    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sw}")

    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    return x.reshape(-1), sr


def _default_chords(bars: int, bpm: float, beats_per_bar: int = 4) -> list[ChordEvent]:
    bar_len = (60.0 / bpm) * beats_per_bar
    names = [("C", "maj"), ("G", "maj"), ("A", "min"), ("F", "maj")]
    out: list[ChordEvent] = []
    for i in range(max(1, bars)):
        s = i * bar_len
        e = (i + 1) * bar_len
        root, q = names[i % len(names)]
        out.append(ChordEvent(start_sec=s, end_sec=e, root_note_name=root, quality=q, confidence=0.9))
    return out


def _write_midi(path: Path, events, bpm: float) -> None:
    import mido

    ticks_per_beat = 480
    tps = ticks_per_beat * (bpm / 60.0)
    tempo = mido.bpm2tempo(bpm)

    timeline = []
    for ev in events:
        timeline.append((ev.time_sec, "note_on", ev.midi_note, ev.velocity))
        timeline.append((ev.time_sec + ev.duration_sec, "note_off", ev.midi_note, 0))
    timeline.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    tr.append(mido.MetaMessage("track_name", name="SOLOMUSE Smoke Lead", time=0))
    tr.append(mido.Message("program_change", program=29, channel=0, time=0))

    last_tick = 0
    for t, kind, note, vel in timeline:
        tick = int(round(t * tps))
        dt = max(0, tick - last_tick)
        tr.append(mido.Message(kind, note=int(note), velocity=int(vel), time=dt))
        last_tick = tick

    mid.save(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke improv generator")
    parser.add_argument("--bpm", type=float, default=110.0)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-wav", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/smoke_improv/output.mid")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpm = float(args.bpm)
    chords: list[ChordEvent]

    if args.input_wav:
        audio, sr = _load_wav_mono(Path(args.input_wav))
        bpm = estimate_bpm(audio, sr, previous_bpm=bpm)
        beats = estimate_beat_times(audio, sr)
        chords = detect_chords_over_time(audio, sr, frame_sec=0.5, beat_times=beats)
        if not chords:
            chords = _default_chords(args.bars, bpm)
    else:
        chords = _default_chords(args.bars, bpm)

    events = generate_improv_events(bpm, chords, play_bars=args.bars, seed=args.seed)

    try:
        _write_midi(out_path, events, bpm)
        print(f"wrote MIDI: {out_path}")
    except Exception:
        json_path = out_path.with_suffix(".json")
        payload = [
            {
                "time_sec": e.time_sec,
                "midi_note": e.midi_note,
                "velocity": e.velocity,
                "duration_sec": e.duration_sec,
            }
            for e in events
        ]
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote JSON fallback: {json_path}")


if __name__ == "__main__":
    main()
