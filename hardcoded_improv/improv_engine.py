from __future__ import annotations

from dataclasses import dataclass
import logging
import random

import numpy as np

from hardcoded_improv.bayes_model import BayesianNoteModel, NoteContext
from hardcoded_improv.chord_detector import ChordEvent
from hardcoded_improv.scale_utils import note_name_to_pc, pentatonic_notes

logger = logging.getLogger(__name__)


@dataclass
class NoteEvent:
    time_sec: float
    midi_note: int
    velocity: int
    duration_sec: float


@dataclass
class HumanizeConfig:
    swing: float = 0.0
    jitter_ms: float = 10.0
    vel_jitter: int = 6
    phrase_len_bars: int = 2
    phrase_gap_prob: float = 0.25
    staccato_prob: float = 0.25
    legato_prob: float = 0.25
    min_dur_frac: float = 0.35
    max_dur_frac: float = 0.95


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


def _apply_swing_and_jitter(
    t: float,
    step: float,
    swing: float,
    jitter_ms: float,
    rng: random.Random,
) -> float:
    step_idx = int(round(t / step)) if step > 0 else 0
    swing_clamped = max(0.0, min(0.35, float(swing)))
    swing_delay = step * swing_clamped if step_idx % 2 == 1 else 0.0
    jitter_s = (max(0.0, float(jitter_ms)) / 1000.0) * rng.uniform(-1.0, 1.0)
    return float(t + swing_delay + jitter_s)


def _velocity_with_accents(
    i: int,
    beats_per_bar: int,
    base_vel: int,
    rng: random.Random,
    phrase_pos: int,
    phrase_len_steps: int,
    vel_jitter: int,
) -> int:
    steps_per_beat = 2
    beat_in_bar = (i // steps_per_beat) % beats_per_bar
    is_offbeat = i % 2 == 1

    vel = int(base_vel)
    if beat_in_bar == 0 and not is_offbeat:
        vel += 12
    elif beats_per_bar >= 4 and beat_in_bar == 2 and not is_offbeat:
        vel += 6

    if is_offbeat:
        vel -= 6

    if phrase_len_steps > 1:
        pos = float(max(0, min(phrase_pos, phrase_len_steps - 1)))
        phase = pos / float(phrase_len_steps - 1)
        envelope = 1.0 - abs(2.0 * phase - 1.0)  # triangle 0..1
        vel += int(round(8.0 * envelope - 2.0))

    vj = max(0, int(vel_jitter))
    vel += rng.randint(-vj, vj)
    return int(max(1, min(127, vel)))


def _duration_with_articulation(
    step: float,
    rng: random.Random,
    staccato_prob: float,
    legato_prob: float,
    min_dur_frac: float,
    max_dur_frac: float,
) -> float:
    min_frac = max(0.1, float(min_dur_frac))
    max_frac = min(1.2, float(max_dur_frac))
    if min_frac > max_frac:
        min_frac, max_frac = max_frac, min_frac

    st_p = max(0.0, min(1.0, float(staccato_prob)))
    lg_p = max(0.0, min(1.0, float(legato_prob)))
    r = rng.random()

    if r < st_p:
        frac = rng.uniform(min_frac, min(min_frac + 0.15, max_frac))
    elif r < st_p + lg_p:
        frac = rng.uniform(max(max_frac - 0.12, min_frac), max_frac)
    else:
        lo = max(min_frac, 0.55)
        hi = min(max_frac, 0.88)
        if lo > hi:
            lo, hi = min_frac, max_frac
        frac = rng.uniform(lo, hi)

    return float(max(1e-4, step * frac))


def generate_improv_events(
    bpm: float,
    chord_timeline: list[ChordEvent],
    play_bars: int = 8,
    beats_per_bar: int = 4,
    seed: int | None = None,
    base_octave: int = 5,
    bayes_model: BayesianNoteModel | None = None,
    humanize_config: HumanizeConfig | None = None,
    humanize_debug: bool = False,
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
    hz = humanize_config or HumanizeConfig()

    sec_per_beat = 60.0 / bpm
    step = sec_per_beat / 2.0  # eighth notes
    total_beats = play_bars * beats_per_bar
    total_steps = total_beats * 2
    phrase_len_steps = max(1, int(hz.phrase_len_bars) * beats_per_bar * 2)

    motif = _motif_rng_pattern(rng)
    motif_pos = 0
    events: list[NoteEvent] = []
    prev_degree = 0
    prev_octave_offset = 0
    prev_midi = 60
    prev_dir = 0
    prev_time_h: float | None = None
    rest_count = 0

    boundary_times = {round(ch.start_sec, 3) for ch in chord_timeline}

    for i in range(total_steps):
        t = i * step
        phrase_pos = i % phrase_len_steps
        is_phrase_start = phrase_pos == 0
        is_phrase_tail = phrase_pos >= max(0, phrase_len_steps - 1)

        if is_phrase_start and rng.random() < max(0.0, min(1.0, hz.phrase_gap_prob)):
            rest_count += 1
            continue
        if is_phrase_tail and rng.random() < max(0.0, min(1.0, hz.phrase_gap_prob * 0.6)):
            rest_count += 1
            continue

        chord = _chord_at_time(chord_timeline, t)
        mode = _mode_from_quality(chord.quality)
        scale_pcs = pentatonic_notes(chord.root_note_name, mode=mode)
        root_pc = note_name_to_pc(chord.root_note_name)

        # Occasional rest (about 20%).
        if rng.random() < 0.2:
            rest_count += 1
            continue

        near_boundary = round(t, 3) in boundary_times
        octave = base_octave + rng.choice([-1, 0, 0, 0, 1])
        if near_boundary and rng.random() < 0.75:
            chosen_degree = 0
            chosen_pc = scale_pcs[chosen_degree]
            base_velocity = rng.randint(88, 112)
        else:
            if bayes_model is not None:
                ctx = NoteContext(
                    prev_degree=prev_degree,
                    beat_pos=i % (beats_per_bar * 2),
                    chord_quality=chord.quality,
                    chord_root=root_pc,
                    prev_interval_direction=prev_dir,
                    prev_octave_offset=prev_octave_offset,
                )
                chosen_degree, sampled_oct = bayes_model.sample_next(ctx, rng)
                chosen_degree = int(chosen_degree) % 5
                chosen_pc = scale_pcs[chosen_degree]
                octave = base_octave + sampled_oct
            else:
                idx = motif[motif_pos % len(motif)]
                motif_pos += 1
                chosen_degree = idx % len(scale_pcs)
                chosen_pc = scale_pcs[chosen_degree]
                octave = base_octave + rng.choice([-1, 0, 0, 0, 1])
            base_velocity = rng.randint(70, 105)

        if near_boundary and rng.random() < 0.5:
            octave = base_octave

        t_h = _apply_swing_and_jitter(t, step, hz.swing, hz.jitter_ms, rng)
        t_h = max(0.0, t_h)
        if prev_time_h is not None:
            if t_h < prev_time_h - 0.03 or t_h <= prev_time_h:
                t_h = prev_time_h + 1e-4

        velocity = _velocity_with_accents(
            i=i,
            beats_per_bar=beats_per_bar,
            base_vel=base_velocity,
            rng=rng,
            phrase_pos=phrase_pos,
            phrase_len_steps=phrase_len_steps,
            vel_jitter=hz.vel_jitter,
        )
        note_dur = _duration_with_articulation(
            step=step,
            rng=rng,
            staccato_prob=hz.staccato_prob,
            legato_prob=hz.legato_prob,
            min_dur_frac=hz.min_dur_frac,
            max_dur_frac=hz.max_dur_frac,
        )

        midi_note = _pc_to_midi(chosen_pc, octave=octave)
        midi_note = max(36, min(96, midi_note))

        if midi_note > prev_midi:
            prev_dir = 1
        elif midi_note < prev_midi:
            prev_dir = -1
        else:
            prev_dir = 0

        prev_degree = chosen_degree
        prev_octave_offset = max(-2, min(2, octave - base_octave))
        prev_midi = midi_note
        prev_time_h = t_h

        events.append(
            NoteEvent(
                time_sec=float(t_h),
                midi_note=int(midi_note),
                velocity=int(velocity),
                duration_sec=float(note_dur),
            )
        )

    events.sort(key=lambda e: e.time_sec)

    if humanize_debug and events:
        times = np.asarray([e.time_sec for e in events], dtype=np.float64)
        vels = np.asarray([e.velocity for e in events], dtype=np.float64)
        durs = np.asarray([e.duration_sec for e in events], dtype=np.float64)
        ioi = np.diff(times)
        ioi_mean = float(np.mean(ioi)) if ioi.size else 0.0
        ioi_std = float(np.std(ioi)) if ioi.size else 0.0
        rest_pct = 100.0 * rest_count / float(max(1, total_steps))
        logger.info(
            "Humanize stats: ioi_mean=%.4f ioi_std=%.4f rests=%.1f%% vel_mean=%.2f vel_std=%.2f dur_min=%.4f dur_max=%.4f",
            ioi_mean,
            ioi_std,
            rest_pct,
            float(np.mean(vels)),
            float(np.std(vels)),
            float(np.min(durs)),
            float(np.max(durs)),
        )

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

