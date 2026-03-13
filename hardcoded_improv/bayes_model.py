from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def _k3(prev_degree: int, beat_pos: int, chord_quality: str) -> str:
    return f"{prev_degree}|{beat_pos}|{chord_quality}"


def _k5(prev_degree: int, beat_pos: int, chord_quality: str, chord_root: int, prev_dir: int) -> str:
    return f"{prev_degree}|{beat_pos}|{chord_quality}|{chord_root}|{prev_dir}"


def _ko(prev_oct: int, prev_dir: int, next_degree: int) -> str:
    return f"{prev_oct}|{prev_dir}|{next_degree}"


def _sample_from_probs(probs: np.ndarray, rng: random.Random) -> int:
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += float(p)
        if r <= c:
            return i
    return int(len(probs) - 1)


@dataclass
class NoteContext:
    prev_degree: int
    beat_pos: int
    chord_quality: str
    chord_root: int
    prev_interval_direction: int
    prev_octave_offset: int


class BayesianNoteModel:
    """Lightweight Bayesian CPT model for next pentatonic degree and octave."""

    def __init__(self, laplace: float = 1.0) -> None:
        self.laplace = float(laplace)
        self.degree_cardinality = 5
        self.octave_values = [-2, -1, 0, 1, 2]

        self.degree_probs_k5: dict[str, list[float]] = {}
        self.degree_probs_k3: dict[str, list[float]] = {}
        self.degree_global: list[float] = [1.0 / self.degree_cardinality] * self.degree_cardinality

        self.octave_probs: dict[str, list[float]] = {}
        self.octave_global: list[float] = [1.0 / len(self.octave_values)] * len(self.octave_values)

    def fit(self, samples: list[dict[str, int | str]]) -> None:
        if not samples:
            logger.warning("No samples provided. Model remains uniform.")
            return

        counts_k5: dict[str, np.ndarray] = {}
        counts_k3: dict[str, np.ndarray] = {}
        global_counts = np.zeros(self.degree_cardinality, dtype=np.float64)

        octave_counts: dict[str, np.ndarray] = {}
        octave_global = np.zeros(len(self.octave_values), dtype=np.float64)

        for s in samples:
            prev_degree = int(s["prev_degree"])
            beat_pos = int(s["beat_pos"])
            chord_quality = str(s["chord_quality"])
            chord_root = int(s["chord_root"])
            prev_dir = int(s["prev_interval_direction"])
            prev_oct = int(s.get("prev_octave_offset", 0))

            next_degree = int(s["next_degree"])
            next_oct = int(s.get("next_octave_offset", 0))
            next_oct = max(-2, min(2, next_oct))

            key5 = _k5(prev_degree, beat_pos, chord_quality, chord_root, prev_dir)
            key3 = _k3(prev_degree, beat_pos, chord_quality)

            if key5 not in counts_k5:
                counts_k5[key5] = np.zeros(self.degree_cardinality, dtype=np.float64)
            if key3 not in counts_k3:
                counts_k3[key3] = np.zeros(self.degree_cardinality, dtype=np.float64)

            counts_k5[key5][next_degree] += 1.0
            counts_k3[key3][next_degree] += 1.0
            global_counts[next_degree] += 1.0

            okey = _ko(prev_oct, prev_dir, next_degree)
            if okey not in octave_counts:
                octave_counts[okey] = np.zeros(len(self.octave_values), dtype=np.float64)
            oidx = self.octave_values.index(next_oct)
            octave_counts[okey][oidx] += 1.0
            octave_global[oidx] += 1.0

        def _normalize_counts(counts: np.ndarray) -> list[float]:
            x = counts + self.laplace
            x = x / np.sum(x)
            return [float(v) for v in x]

        self.degree_probs_k5 = {k: _normalize_counts(v) for k, v in counts_k5.items()}
        self.degree_probs_k3 = {k: _normalize_counts(v) for k, v in counts_k3.items()}
        self.degree_global = _normalize_counts(global_counts)

        self.octave_probs = {k: _normalize_counts(v) for k, v in octave_counts.items()}
        self.octave_global = _normalize_counts(octave_global)

        logger.info(
            "Bayes fit: samples=%d k5_states=%d k3_states=%d octave_states=%d",
            len(samples),
            len(self.degree_probs_k5),
            len(self.degree_probs_k3),
            len(self.octave_probs),
        )

        # quick debug: top transitions by mean probability per k3
        top: list[tuple[str, int, float]] = []
        for k, probs in self.degree_probs_k3.items():
            idx = int(np.argmax(probs))
            top.append((k, idx, float(probs[idx])))
        top = sorted(top, key=lambda x: x[2], reverse=True)[:5]
        for k, idx, p in top:
            logger.info("Top transition: context=%s -> degree=%d (p=%.3f)", k, idx, p)

    def sample_next(self, context: NoteContext, rng: random.Random) -> tuple[int, int]:
        key5 = _k5(
            context.prev_degree,
            context.beat_pos,
            context.chord_quality,
            context.chord_root,
            context.prev_interval_direction,
        )
        key3 = _k3(context.prev_degree, context.beat_pos, context.chord_quality)

        if key5 in self.degree_probs_k5:
            degree_probs = np.asarray(self.degree_probs_k5[key5], dtype=np.float64)
        elif key3 in self.degree_probs_k3:
            degree_probs = np.asarray(self.degree_probs_k3[key3], dtype=np.float64)
        else:
            degree_probs = np.asarray(self.degree_global, dtype=np.float64)

        next_degree = _sample_from_probs(degree_probs, rng)

        okey = _ko(context.prev_octave_offset, context.prev_interval_direction, next_degree)
        if okey in self.octave_probs:
            oct_probs = np.asarray(self.octave_probs[okey], dtype=np.float64)
        else:
            oct_probs = np.asarray(self.octave_global, dtype=np.float64)
        oct_idx = _sample_from_probs(oct_probs, rng)
        next_oct = self.octave_values[oct_idx]

        return int(next_degree), int(next_oct)

    def save_json(self, path: str) -> None:
        payload = {
            "laplace": self.laplace,
            "degree_cardinality": self.degree_cardinality,
            "octave_values": self.octave_values,
            "degree_probs_k5": self.degree_probs_k5,
            "degree_probs_k3": self.degree_probs_k3,
            "degree_global": self.degree_global,
            "octave_probs": self.octave_probs,
            "octave_global": self.octave_global,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "BayesianNoteModel":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        model = cls(laplace=float(payload.get("laplace", 1.0)))
        model.degree_cardinality = int(payload.get("degree_cardinality", 5))
        model.octave_values = [int(v) for v in payload.get("octave_values", [-2, -1, 0, 1, 2])]
        model.degree_probs_k5 = {str(k): [float(x) for x in v] for k, v in payload.get("degree_probs_k5", {}).items()}
        model.degree_probs_k3 = {str(k): [float(x) for x in v] for k, v in payload.get("degree_probs_k3", {}).items()}
        model.degree_global = [float(x) for x in payload.get("degree_global", [0.2] * 5)]
        model.octave_probs = {str(k): [float(x) for x in v] for k, v in payload.get("octave_probs", {}).items()}
        model.octave_global = [float(x) for x in payload.get("octave_global", [0.2] * 5)]
        return model
