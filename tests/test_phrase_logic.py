from __future__ import annotations

from collections import defaultdict

import numpy as np

from hardcoded_improv.chord_detector import ChordEvent
from hardcoded_improv.improv_engine import HumanizeConfig, LickConfig, generate_improv_events


def _chords(play_bars: int = 12) -> list[ChordEvent]:
    # Static harmony so degree analysis is stable.
    total = float(play_bars * 2)  # at 120bpm and 4/4 this is enough for timeline windowing
    return [ChordEvent(0.0, total, "C", "maj", 0.9)]


def _pc_to_degree_c_major_pent(pc: int) -> int | None:
    mp = {0: 0, 2: 1, 4: 2, 7: 3, 9: 4}
    return mp.get(pc % 12)


def _edit_distance(a: list[int], b: list[int]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def test_strong_beats_target_more_than_weak() -> None:
    bpm = 120.0
    step = 60.0 / bpm / 2.0
    hz = HumanizeConfig(swing=0.0, jitter_ms=0.0, phrase_gap_prob=0.0)
    lk = LickConfig(lick_prob_on_boundary=0.0, lick_prob_on_phrase_start=0.0, grace_note_prob=0.0, slide_prob=0.0)

    events = generate_improv_events(
        bpm,
        _chords(play_bars=16),
        play_bars=16,
        seed=314,
        humanize_config=hz,
        lick_config=lk,
    )

    strong_hits = 0
    strong_total = 0
    weak_hits = 0
    weak_total = 0

    target_degrees = {0, 2, 4}  # maj-mode targets in implementation

    for ev in events:
        idx = int(round(ev.time_sec / step))
        beat_in_bar = (idx // 2) % 4
        is_strong = idx % 2 == 0 and beat_in_bar in {0, 2}
        deg = _pc_to_degree_c_major_pent(ev.midi_note % 12)
        if deg is None:
            continue
        if is_strong:
            strong_total += 1
            if deg in target_degrees:
                strong_hits += 1
        else:
            weak_total += 1
            if deg in target_degrees:
                weak_hits += 1

    assert strong_total > 10 and weak_total > 10
    strong_rate = strong_hits / strong_total
    weak_rate = weak_hits / weak_total
    assert strong_rate > weak_rate


def test_call_response_similar_not_identical() -> None:
    bpm = 120.0
    step = 60.0 / bpm / 2.0
    phrase_len_steps = 2 * 4 * 2  # phrase_len_bars=2, beats_per_bar=4, eighth grid

    hz = HumanizeConfig(swing=0.0, jitter_ms=0.0, phrase_gap_prob=0.0, phrase_len_bars=2)
    lk = LickConfig(lick_prob_on_boundary=0.0, lick_prob_on_phrase_start=0.0, grace_note_prob=0.0, slide_prob=0.0)

    events = generate_improv_events(
        bpm,
        _chords(play_bars=8),
        play_bars=8,
        seed=21,
        humanize_config=hz,
        lick_config=lk,
    )

    steps_to_deg: dict[int, int] = {}
    for ev in events:
        idx = int(round(ev.time_sec / step))
        deg = _pc_to_degree_c_major_pent(ev.midi_note % 12)
        if deg is not None:
            steps_to_deg[idx] = deg

    p0 = [steps_to_deg.get(i, -1) for i in range(0, phrase_len_steps)]
    p1 = [steps_to_deg.get(i, -1) for i in range(phrase_len_steps, phrase_len_steps * 2)]

    dist = _edit_distance(p0, p1)
    assert dist > 0
    assert dist < int(0.75 * phrase_len_steps)


def test_range_window_respected_except_phrase_boundary() -> None:
    bpm = 120.0
    step = 60.0 / bpm / 2.0
    phrase_len_steps = 2 * 4 * 2

    hz = HumanizeConfig(swing=0.0, jitter_ms=0.0, phrase_gap_prob=0.0, phrase_len_bars=2)
    lk = LickConfig(lick_prob_on_boundary=0.0, lick_prob_on_phrase_start=0.0, grace_note_prob=0.0, slide_prob=0.0)

    events = generate_improv_events(
        bpm,
        _chords(play_bars=10),
        play_bars=10,
        seed=77,
        humanize_config=hz,
        lick_config=lk,
    )

    by_step = defaultdict(list)
    for ev in events:
        idx = int(round(ev.time_sec / step))
        by_step[idx].append(ev)

    main_events = []
    for idx in sorted(by_step.keys()):
        ev = max(by_step[idx], key=lambda e: e.velocity)
        main_events.append((idx, ev.midi_note))

    for (idx_a, n_a), (idx_b, n_b) in zip(main_events, main_events[1:]):
        jump = abs(n_b - n_a)
        boundary = (idx_b % phrase_len_steps) == 0
        if not boundary:
            assert jump <= 9
