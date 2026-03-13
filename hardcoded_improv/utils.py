from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    return 20.0 * np.log10(rms)


def save_wav_mono(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2 and mono.shape[1] == 1:
        mono = mono[:, 0]
    mono = mono.reshape(-1)

    pcm16 = np.int16(np.clip(mono, -1.0, 1.0) * 32767.0)

    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())

    return out_path
