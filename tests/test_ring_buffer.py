from __future__ import annotations

import wave

import numpy as np

from hardcoded_improv.ring_buffer import ThreadSafeRingBuffer
from hardcoded_improv.utils import save_wav_mono


def test_ring_buffer_wraps_and_returns_last_samples() -> None:
    rb = ThreadSafeRingBuffer(capacity_samples=8, channels=1)

    rb.write(np.arange(0, 5, dtype=np.float32))
    rb.write(np.arange(5, 12, dtype=np.float32))

    out = rb.get_last(8)
    expected = np.arange(4, 12, dtype=np.float32)[:, None]
    assert out.shape == (8, 1)
    np.testing.assert_allclose(out, expected)


def test_get_last_seconds() -> None:
    rb = ThreadSafeRingBuffer(capacity_samples=100, channels=1)
    rb.write(np.arange(0, 100, dtype=np.float32))

    out = rb.get_last_seconds(0.5, sample_rate=100)
    expected = np.arange(50, 100, dtype=np.float32)[:, None]
    assert out.shape == (50, 1)
    np.testing.assert_allclose(out, expected)


def test_save_wav_mono(tmp_path) -> None:
    sr = 16000
    audio = np.linspace(-0.5, 0.5, sr, dtype=np.float32)
    out_path = tmp_path / "debug.wav"

    save_wav_mono(out_path, audio, sr)

    assert out_path.exists()
    with wave.open(str(out_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == sr
        assert wf.getnframes() == sr
