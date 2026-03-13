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


@dataclass
class LickConfig:
    lick_prob_on_boundary: float = 0.35
    lick_prob_on_phrase_start: float = 0.45
    grace_note_prob: float = 0.20
    slide_prob: float = 0.10
    max_lick_len_steps: int = 6
    use_pitch_bend: bool = False


LICK_TEMPLATES: dict[str, list[list[int]]] = {
    "maj": [
        [0, 2, 1, 3],
        [2, 1, 0, 1],
        [0, 1, 2, 1],
        [0, 2, 0],
        [1, 3, 2, 4],
    ],
    "min": [
        [0, 1, 2, 1],
        [2, 1, 0, 1],
        [0, 2, 1, 3],
        [0, 2, 0],
        [3, 2, 1, 0],
    ],
}


def _target_degrees_for_mode(mode: str) -> set[int]:
    if mode == "minor":
        return {0, 2, 3}
    return {0, 2, 4}


def _generate_call_motif(rng: random.Random, start_degree: int) -> list[int]:
    length = rng.randint(4, 6)
    out = [int(max(0, min(4, start_degree)))]
    for _ in range(length - 1):
        step = rng.choice([-1, 0, 1, 1])
        out.append(int(max(0, min(4, out[-1] + step))))
    return out


def _mutate_response_motif(base: list[int], rng: random.Random) -> list[int]:
    if not base:
        return _generate_call_motif(rng, start_degree=0)
    m = list(base)
    idx = rng.randrange(len(m))
    m[idx] = int(max(0, min(4, m[idx] + rng.choice([-1, 1]))))
    if rng.random() < 0.35:
        m = list(reversed(m))
    return m


def _stepwise_degree(prev_degree: int, rng: random.Random) -> int:
    return int(max(0, min(4, prev_degree + rng.choice([-1, 0, 1]))))


def _clamp_jump_by_octave(midi_note: int, prev_midi: int, max_jump: int = 9) -> int:
    n = int(midi_note)
    if abs(n - prev_midi) <= max_jump:
        return n

    while n - prev_midi > max_jump and n - 12 >= 36:
        n -= 12
    while prev_midi - n > max_jump and n + 12 <= 96:
        n += 12

    if n - prev_midi > max_jump:
        n = prev_midi + max_jump
    elif prev_midi - n > max_jump:
        n = prev_midi - max_jump

    return int(max(36, min(96, n)))


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


def _pick_lick_template(
    mode: str,
    prev_dir: int,
    beat_pos: int,
    beats_per_bar: int,
    max_len: int,
    rng: random.Random,
) -> list[int]:
    pool = LICK_TEMPLATES.get(mode, LICK_TEMPLATES["maj"])

    strong_positions = {0}
    if beats_per_bar >= 4:
        strong_positions.add((beats_per_bar // 2) * 2)
    is_strong = beat_pos in strong_positions

    candidates = pool
    if prev_dir > 0:
        candidates = [p for p in pool if p[-1] >= p[0]] or pool
    elif prev_dir < 0:
        candidates = [p for p in pool if p[-1] <= p[0]] or pool

    if is_strong:
        candidates = sorted(candidates, key=len, reverse=True)

    chosen = list(rng.choice(candidates))
    max_l = max(1, int(max_len))
    chosen = chosen[:max_l]
    return [int(max(0, min(4, d))) for d in chosen]


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
    lick_config: LickConfig | None = None,
    lick_debug: bool = False,
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
    lk = lick_config or LickConfig()

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
    current_lick: list[int] = []
    call_motif: list[int] = []
    active_phrase_motif: list[int] = []
    last_phrase_id = -1
    pending_target_step: int | None = None
    pending_target_degree: int | None = None
    max_jump_semitones = 9

    boundary_times = {round(ch.start_sec, 3) for ch in chord_timeline}

    for i in range(total_steps):
        t = i * step
        phrase_id = i // phrase_len_steps
        phrase_pos = i % phrase_len_steps
        is_phrase_start = phrase_pos == 0
        is_phrase_tail = phrase_pos >= max(0, phrase_len_steps - 1)

        if phrase_id != last_phrase_id:
            if phrase_id % 2 == 0:
                call_motif = _generate_call_motif(rng, start_degree=prev_degree)
                active_phrase_motif = list(call_motif)
            else:
                active_phrase_motif = _mutate_response_motif(call_motif, rng)
            last_phrase_id = phrase_id

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
        beat_pos = i % (beats_per_bar * 2)
        beat_in_bar = (i // 2) % beats_per_bar
        is_onbeat = i % 2 == 0
        is_strong_beat = is_onbeat and (beat_in_bar == 0 or (beats_per_bar >= 4 and beat_in_bar == 2))
        next_step_is_strong = ((i + 1) % 2 == 0) and (((i + 1) // 2) % beats_per_bar in {0, 2 if beats_per_bar >= 4 else 0})

        target_degrees = _target_degrees_for_mode(mode)
        phrase_hint = active_phrase_motif[phrase_pos % len(active_phrase_motif)] if active_phrase_motif else prev_degree

        if not current_lick:
            p = 0.0
            if near_boundary:
                p = max(p, float(lk.lick_prob_on_boundary))
            if is_phrase_start:
                p = max(p, float(lk.lick_prob_on_phrase_start))

            if p > 0.0 and rng.random() < min(1.0, p):
                current_lick = _pick_lick_template(
                    mode=mode,
                    prev_dir=prev_dir,
                    beat_pos=beat_pos,
                    beats_per_bar=beats_per_bar,
                    max_len=lk.max_lick_len_steps,
                    rng=rng,
                )
                if lick_debug:
                    logger.info(
                        "Lick start: template=%s step=%d chord=%s:%s mode=%s",
                        current_lick,
                        i,
                        chord.root_note_name,
                        chord.quality,
                        mode,
                    )

        chosen_from_lick = False
        if near_boundary and rng.random() < 0.75:
            chosen_degree = 0
            chosen_pc = scale_pcs[chosen_degree]
            base_velocity = rng.randint(88, 112)
        else:
            if current_lick:
                chosen_degree = int(current_lick.pop(0))
                chosen_degree = max(0, min(4, chosen_degree))
                chosen_pc = scale_pcs[chosen_degree]
                chosen_from_lick = True
                base_velocity = rng.randint(74, 108)
            elif pending_target_step is not None and pending_target_degree is not None and i == pending_target_step:
                chosen_degree = int(max(0, min(4, pending_target_degree)))
                chosen_pc = scale_pcs[chosen_degree]
                pending_target_step = None
                pending_target_degree = None
                base_velocity = rng.randint(82, 110)
            elif is_strong_beat and rng.random() < 0.78:
                if phrase_hint in target_degrees and rng.random() < 0.65:
                    chosen_degree = int(phrase_hint)
                else:
                    chosen_degree = int(rng.choice(sorted(target_degrees)))
                chosen_pc = scale_pcs[chosen_degree]
                base_velocity = rng.randint(80, 110)
            elif (not is_strong_beat) and next_step_is_strong and rng.random() < 0.68:
                upcoming_target = int(rng.choice(sorted(target_degrees)))
                pending_target_step = i + 1
                pending_target_degree = upcoming_target
                if upcoming_target <= 0:
                    chosen_degree = 1
                elif upcoming_target >= 4:
                    chosen_degree = 3
                else:
                    chosen_degree = int(upcoming_target + rng.choice([-1, 1]))
                chosen_degree = max(0, min(4, chosen_degree))
                chosen_pc = scale_pcs[chosen_degree]
                base_velocity = rng.randint(68, 100)
            elif bayes_model is not None:
                ctx = NoteContext(
                    prev_degree=prev_degree,
                    beat_pos=beat_pos,
                    chord_quality=chord.quality,
                    chord_root=root_pc,
                    prev_interval_direction=prev_dir,
                    prev_octave_offset=prev_octave_offset,
                )
                chosen_degree, sampled_oct = bayes_model.sample_next(ctx, rng)
                chosen_degree = int(chosen_degree) % 5
                chosen_pc = scale_pcs[chosen_degree]
                octave = base_octave + sampled_oct
                base_velocity = rng.randint(70, 105)
            else:
                if rng.random() < 0.65:
                    chosen_degree = _stepwise_degree(prev_degree, rng)
                else:
                    idx = motif[motif_pos % len(motif)]
                    motif_pos += 1
                    chosen_degree = idx % len(scale_pcs)

                if active_phrase_motif and rng.random() < 0.40:
                    chosen_degree = int(max(0, min(4, phrase_hint)))

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

        if not is_phrase_start:
            midi_note = _clamp_jump_by_octave(midi_note, prev_midi, max_jump=max_jump_semitones)
        elif is_phrase_tail and rng.random() < 0.20:
            midi_note = int(max(36, min(96, midi_note + 12)))

        # Ornament layer (kept simple for MIDI note-only pipeline).
        inserted_ornament = False
        slide_prob = max(0.0, min(1.0, float(lk.slide_prob)))
        grace_prob = max(0.0, min(1.0, float(lk.grace_note_prob)))

        if rng.random() < slide_prob:
            if lk.use_pitch_bend:
                # TODO: add pitch-bend event structure in MIDI writer and emit ramp here.
                if lick_debug:
                    logger.info("Slide requested with pitch-bend path at step=%d (TODO, currently no-op)", i)
            else:
                a_time = max(0.0, t_h - 0.08)
                if prev_time_h is None or a_time >= prev_time_h:
                    approach = max(36, min(96, midi_note - 1))
                    a_vel = max(1, min(127, velocity - 10))
                    a_dur = max(0.015, min(0.06, note_dur * 0.3))
                    events.append(
                        NoteEvent(
                            time_sec=float(a_time),
                            midi_note=int(approach),
                            velocity=int(a_vel),
                            duration_sec=float(a_dur),
                        )
                    )
                    inserted_ornament = True

        if (not inserted_ornament) and rng.random() < grace_prob:
            grace_offset = rng.uniform(0.03, 0.06)
            g_time = max(0.0, t_h - grace_offset)
            if prev_time_h is None or g_time >= prev_time_h:
                step_dir = -1 if rng.random() < 0.65 else 1
                grace_note = max(36, min(96, midi_note + step_dir * rng.choice([1, 2])))
                g_vel = max(1, min(127, velocity - rng.randint(6, 14)))
                g_dur = max(0.012, min(0.05, note_dur * 0.25))
                events.append(
                    NoteEvent(
                        time_sec=float(g_time),
                        midi_note=int(grace_note),
                        velocity=int(g_vel),
                        duration_sec=float(g_dur),
                    )
                )
                if lick_debug:
                    logger.info("Grace note inserted: step=%d time=%.3f note=%d -> %d", i, g_time, grace_note, midi_note)

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

