from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from hardcoded_improv.improv_engine import NoteEvent

logger = logging.getLogger(__name__)

try:
    import mido
except Exception:  # pragma: no cover - runtime optional dependency path
    mido = None


def list_midi_output_ports() -> list[str]:
    if mido is None:
        return []
    return [str(x) for x in mido.get_output_names()]


@dataclass
class ScheduledMidiEvent:
    time_sec: float
    kind: str
    note: int
    velocity: int


class MidiOut:
    def __init__(self, port_name: str | None = None) -> None:
        if mido is None:
            raise RuntimeError("mido is not available. Install mido + python-rtmidi.")

        names = mido.get_output_names()
        if port_name is not None:
            self._port = mido.open_output(port_name)
            self.port_name = port_name
        elif names:
            self._port = mido.open_output(names[0])
            self.port_name = names[0]
        else:
            raise RuntimeError("No MIDI output ports available")

        logger.info("Opened MIDI output: %s", self.port_name)

    def send_note_on(self, note: int, velocity: int, channel: int = 0) -> None:
        self._port.send(mido.Message("note_on", note=int(note), velocity=int(velocity), channel=channel))

    def send_note_off(self, note: int, channel: int = 0) -> None:
        self._port.send(mido.Message("note_off", note=int(note), velocity=0, channel=channel))

    def close(self) -> None:
        self._port.close()

    def __enter__(self) -> "MidiOut":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()


def build_scheduled_events(events: list[NoteEvent]) -> list[ScheduledMidiEvent]:
    scheduled: list[ScheduledMidiEvent] = []
    for ev in events:
        scheduled.append(
            ScheduledMidiEvent(
                time_sec=ev.time_sec,
                kind="on",
                note=ev.midi_note,
                velocity=ev.velocity,
            )
        )
        scheduled.append(
            ScheduledMidiEvent(
                time_sec=ev.time_sec + ev.duration_sec,
                kind="off",
                note=ev.midi_note,
                velocity=0,
            )
        )
    scheduled.sort(key=lambda e: (e.time_sec, 0 if e.kind == "off" else 1))
    return scheduled


def play_events_realtime(
    events: list[NoteEvent],
    midi_out: MidiOut | None = None,
    start_monotonic: float | None = None,
    dry_run: bool = False,
) -> None:
    if not events:
        return

    timeline = build_scheduled_events(events)
    t0 = time.monotonic() if start_monotonic is None else start_monotonic

    for item in timeline:
        target = t0 + item.time_sec
        while True:
            now = time.monotonic()
            dt = target - now
            if dt <= 0:
                break
            time.sleep(min(0.002, dt))

        if dry_run:
            logger.info("[dry-run] t=%.3f kind=%s note=%d vel=%d", item.time_sec, item.kind, item.note, item.velocity)
            continue

        if midi_out is None:
            raise ValueError("midi_out is required when dry_run=False")

        if item.kind == "on":
            midi_out.send_note_on(item.note, item.velocity)
        else:
            midi_out.send_note_off(item.note)
