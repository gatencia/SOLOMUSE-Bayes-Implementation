from __future__ import annotations

import json
import wave

import numpy as np

from hardcoded_improv.config import AppConfig
from hardcoded_improv.live_demo import run_simulation_demo


def _write_test_wav(path: str, sr: int = 22050, duration_s: float = 10.0) -> None:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr

    # C major chord bed.
    audio = (
        0.20 * np.sin(2 * np.pi * 261.63 * t)
        + 0.18 * np.sin(2 * np.pi * 329.63 * t)
        + 0.16 * np.sin(2 * np.pi * 392.00 * t)
    ).astype(np.float32)

    # Add beat clicks at 120 BPM.
    beat_period = 0.5
    click_len = int(0.01 * sr)
    click = np.hanning(max(4, click_len)).astype(np.float32)
    cur = 0.0
    while cur < duration_s:
        idx = int(cur * sr)
        end = min(n, idx + click_len)
        seg = end - idx
        if seg > 0:
            audio[idx:end] += 0.8 * click[:seg]
        cur += beat_period

    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def test_simulation_pipeline_produces_artifacts(tmp_path) -> None:
    wav_path = tmp_path / "input.wav"
    _write_test_wav(str(wav_path), sr=22050, duration_s=10.0)

    artifacts = tmp_path / "artifacts"
    cfg = AppConfig(sample_rate=22050, blocksize=512, channels=1, ring_buffer_seconds=20)

    result = run_simulation_demo(
        cfg,
        input_wav=wav_path,
        listen_bars=2,
        play_bars=4,
        artifacts_dir=artifacts,
        seed=7,
        output_mid=True,
    )

    assert result.bpm > 0
    assert (artifacts / "listen_audio.wav").exists()
    assert (artifacts / "chords.json").exists()
    assert (artifacts / "events.csv").exists()
    assert (artifacts / "output.mid").exists()

    with (artifacts / "chords.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert float(payload["bpm"]) > 0
    assert len(payload["chords"]) > 0

    lines = (artifacts / "events.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 1
