from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from pathlib import Path
import wave

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

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


@dataclass
class PhraseConfig:
    phrase_len_bars: int = 2
    enable_call_response: bool = True
    strong_target_prob: float = 0.78
    approach_prob: float = 0.68
    max_jump_semitones: int = 9


@dataclass
class GrooveConfig:
    enabled: bool = True
    lock_strength: float = 0.7
    density_influence: float = 0.55
    max_offset_sec: float = 0.08
    base_play_prob: float = 0.08
    density_play_scale: float = 0.85
    strong_beat_bonus: float = 0.22
    min_density_gate: float = 0.12


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


def _subdiv_positions_16th(swing_ratio: float) -> list[float]:
    r = float(max(0.5, min(0.66, swing_ratio)))
    return [0.0, 0.5 * r, r, r + 0.5 * (1.0 - r)]


def _load_wav_mono_float32(path: str | Path) -> tuple[np.ndarray, int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"WAV not found: {p}")
    with wave.open(str(p), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        n = int(wf.getnframes())
        raw = wf.readframes(n)

    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sw}")

    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    return x.reshape(-1), sr


def build_groove_grid_from_audio(
    audio: np.ndarray,
    sr: int,
    swing_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return groove-aligned 16th grid times, onset scores, beat times, and BPM.

    Grid is derived from beat intervals and optionally swung with `swing_ratio`.
    """
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), 120.0

    if librosa is None:
        dur = y.size / float(sr)
        bpm = 120.0
        beat_period = 60.0 / bpm
        beats = np.arange(0.0, dur + beat_period, beat_period, dtype=np.float32)
        grid = np.arange(0.0, dur, beat_period / 4.0, dtype=np.float32)
        scores = np.ones_like(grid, dtype=np.float32) * 0.5
        return grid, scores, beats, bpm

    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=2048)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop, trim=False)
    tempo_arr = np.asarray(tempo, dtype=np.float64).reshape(-1)
    tempo_val = float(tempo_arr[0]) if tempo_arr.size else float("nan")
    bpm = tempo_val if np.isfinite(tempo_val) and tempo_val > 0 else 120.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop).astype(np.float32)

    if beat_times.size < 2:
        beat_period = 60.0 / bpm
        dur = y.size / float(sr)
        beat_times = np.arange(0.0, dur + beat_period, beat_period, dtype=np.float32)

    pos = _subdiv_positions_16th(swing_ratio)
    grid_times: list[float] = []
    for i in range(len(beat_times) - 1):
        t0 = float(beat_times[i])
        t1 = float(beat_times[i + 1])
        if t1 <= t0:
            continue
        d = t1 - t0
        for p in pos:
            grid_times.append(t0 + p * d)

    grid = np.asarray(grid_times, dtype=np.float32)
    grid = grid[(grid >= 0.0) & (grid <= (y.size / float(sr)) + 1e-6)]
    if grid.size == 0:
        return grid, np.array([], dtype=np.float32), beat_times, bpm

    frame_times = librosa.frames_to_time(np.arange(onset_env.size), sr=sr, hop_length=hop)
    idx = np.searchsorted(frame_times, grid, side="left")
    idx = np.clip(idx, 0, max(0, onset_env.size - 1))
    scores = onset_env[idx].astype(np.float32)
    if scores.size:
        mn, mx = float(np.min(scores)), float(np.max(scores))
        if mx > mn + 1e-9:
            scores = (scores - mn) / (mx - mn)
        else:
            scores = np.zeros_like(scores) + 0.5

    logger.info(
        "Groove grid: bpm=%.2f beats=%d grid=%d swing=%.2f onset_mean=%.3f",
        bpm,
        int(beat_times.size),
        int(grid.size),
        float(max(0.5, min(0.66, swing_ratio))),
        float(np.mean(scores)) if scores.size else 0.0,
    )
    return grid, scores, beat_times, bpm


def build_groove_grid_from_wav(
    wav_path: str | Path,
    swing_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    audio, sr = _load_wav_mono_float32(wav_path)
    return build_groove_grid_from_audio(audio, sr, swing_ratio=swing_ratio)


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
    humanize: HumanizeConfig | None = None,
    lick_cfg: LickConfig | None = None,
    phrase_cfg: PhraseConfig | None = None,
    groove_cfg: GrooveConfig | None = None,
    groove_offsets: list[float] | None = None,
    groove_density: list[float] | None = None,
    groove_grid_times: list[float] | np.ndarray | None = None,
    groove_onset_scores: list[float] | np.ndarray | None = None,
    groove_beat_times: list[float] | np.ndarray | None = None,
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

    # New preferred params + backward-compatible aliases.
    hz_in = humanize if humanize is not None else humanize_config
    lk_in = lick_cfg if lick_cfg is not None else lick_config
    ph_in = phrase_cfg
    gr_in = groove_cfg

    # "old simple style" defaults per-module when not explicitly provided.
    no_configs = hz_in is None and lk_in is None and ph_in is None and gr_in is None
    if no_configs:
        hz = HumanizeConfig(
            swing=0.0,
            jitter_ms=0.0,
            vel_jitter=0,
            phrase_len_bars=2,
            phrase_gap_prob=0.0,
            staccato_prob=0.0,
            legato_prob=0.0,
            min_dur_frac=0.85,
            max_dur_frac=0.85,
        )
        lk = LickConfig(
            lick_prob_on_boundary=0.0,
            lick_prob_on_phrase_start=0.0,
            grace_note_prob=0.0,
            slide_prob=0.0,
            max_lick_len_steps=6,
            use_pitch_bend=False,
        )
        ph = PhraseConfig(
            phrase_len_bars=2,
            enable_call_response=False,
            strong_target_prob=0.0,
            approach_prob=0.0,
            max_jump_semitones=12,
        )
        gr = GrooveConfig(enabled=False, lock_strength=0.0, density_influence=0.0, max_offset_sec=0.0)
    else:
        hz = hz_in or HumanizeConfig()
        lk = lk_in or LickConfig()
        ph = ph_in or PhraseConfig(phrase_len_bars=hz.phrase_len_bars)
        gr = gr_in or GrooveConfig(enabled=False, lock_strength=0.0, density_influence=0.0, max_offset_sec=0.0)

    sec_per_beat = 60.0 / bpm
    step = sec_per_beat / 2.0  # eighth notes
    total_beats = play_bars * beats_per_bar
    total_duration = total_beats * sec_per_beat

    if groove_grid_times is not None:
        g = np.asarray(groove_grid_times, dtype=np.float32).reshape(-1)
        g = g[(g >= 0.0) & (g < total_duration + 1e-6)]
        candidate_times = [float(x) for x in g]
    else:
        candidate_times = [i * step for i in range(total_beats * 2)]

    if not candidate_times:
        return []

    total_steps = len(candidate_times)
    phrase_len_steps = max(1, int(ph.phrase_len_bars) * beats_per_bar * 2)

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
    max_jump_semitones = max(1, int(ph.max_jump_semitones))
    bar_len = sec_per_beat * beats_per_bar
    notes_per_bar: dict[int, int] = {}
    cooldown = 0

    boundary_times = {round(ch.start_sec, 3) for ch in chord_timeline}
    chord_changes = sorted({float(ch.start_sec) for ch in chord_timeline if ch.start_sec > 0.0})

    onset_scores = None
    if groove_onset_scores is not None:
        arr = np.asarray(groove_onset_scores, dtype=np.float32).reshape(-1)
        if arr.size >= total_steps:
            onset_scores = arr

    for i in range(total_steps):
        t = float(candidate_times[i])
        phrase_id = i // phrase_len_steps
        phrase_pos = i % phrase_len_steps
        is_phrase_start = phrase_pos == 0
        is_phrase_tail = phrase_pos >= max(0, phrase_len_steps - 1)

        if cooldown > 0:
            cooldown -= 1
            rest_count += 1
            continue

        if phrase_id != last_phrase_id:
            if phrase_id % 2 == 0:
                call_motif = _generate_call_motif(rng, start_degree=prev_degree)
                active_phrase_motif = list(call_motif)
            else:
                if ph.enable_call_response:
                    active_phrase_motif = _mutate_response_motif(call_motif, rng)
                else:
                    active_phrase_motif = list(call_motif)
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

        slot_count = beats_per_bar * 2
        beat_float = t / sec_per_beat
        beat_in_bar_float = beat_float % beats_per_bar
        slot = int(round(beat_in_bar_float * 2.0)) % slot_count

        bar_idx = int(t // bar_len) if bar_len > 1e-6 else 0
        if notes_per_bar.get(bar_idx, 0) >= 8:
            rest_count += 1
            continue

        near_boundary = round(t, 3) in boundary_times
        octave = base_octave + rng.choice([-1, 0, 0, 0, 1])
        beat_pos = slot
        beat_in_bar = int(beat_in_bar_float)
        is_onbeat = slot % 2 == 0
        is_strong_beat = is_onbeat and (beat_in_bar == 0 or (beats_per_bar >= 4 and beat_in_bar == 2))
        next_step_is_strong = (slot % 2 == 1) and ((beat_in_bar in {0, 2} if beats_per_bar >= 4 else beat_in_bar == 0))

        next_change = next((c for c in chord_changes if c >= t), None)
        near_change = next_change is not None and (next_change - t) <= 0.20
        bar_phase = t % bar_len if bar_len > 1e-6 else 0.0
        near_bar_boundary = bar_phase <= 0.10 or (bar_len - bar_phase) <= 0.10

        # Groove-gated note density: only play where source groove has energy.
        play_prob = 1.0 - 0.2  # fallback legacy behavior when groove disabled
        if gr.enabled and groove_density is not None and len(groove_density) >= slot_count:
            den = float(max(0.0, min(1.0, groove_density[slot])))
            if den < float(gr.min_density_gate):
                rest_count += 1
                continue
            play_prob = float(gr.base_play_prob) + float(gr.density_play_scale) * den
            if is_strong_beat:
                play_prob += float(gr.strong_beat_bonus)
            play_prob = max(0.02, min(0.98, play_prob))

        if onset_scores is not None and i < onset_scores.size:
            osc = float(max(0.0, min(1.0, onset_scores[i])))
            play_prob = 0.15 + 0.75 * osc + 0.10 * play_prob
            play_prob = max(0.02, min(0.98, play_prob))

        if rng.random() > play_prob:
            rest_count += 1
            continue

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
        if (near_boundary or near_change or near_bar_boundary) and rng.random() < 0.92:
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
            elif is_strong_beat and rng.random() < max(0.0, min(1.0, ph.strong_target_prob)):
                if phrase_hint in target_degrees and rng.random() < 0.65:
                    chosen_degree = int(phrase_hint)
                else:
                    chosen_degree = int(rng.choice(sorted(target_degrees)))
                chosen_pc = scale_pcs[chosen_degree]
                base_velocity = rng.randint(80, 110)
            elif (not is_strong_beat) and next_step_is_strong and rng.random() < max(0.0, min(1.0, ph.approach_prob)):
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

        t_base = t
        if gr.enabled and groove_offsets is not None and len(groove_offsets) >= slot_count:
            off = float(groove_offsets[slot])
            off = max(-float(gr.max_offset_sec), min(float(gr.max_offset_sec), off))
            t_base = t + off * max(0.0, min(1.0, float(gr.lock_strength)))

        t_h = _apply_swing_and_jitter(t_base, step, hz.swing, hz.jitter_ms, rng)
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
        notes_per_bar[bar_idx] = notes_per_bar.get(bar_idx, 0) + 1
        if note_dur > step * 0.86:
            cooldown = max(cooldown, 1)

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

    if (humanize_debug or lick_debug) and notes_per_bar:
        dens = ", ".join([f"bar{b}:{c}" for b, c in sorted(notes_per_bar.items())])
        logger.info("Note density per bar: %s", dens)

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

