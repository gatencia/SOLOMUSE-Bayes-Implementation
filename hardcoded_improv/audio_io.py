from __future__ import annotations

import logging
from threading import Lock

import numpy as np
import sounddevice as sd

from hardcoded_improv.config import AppConfig
from hardcoded_improv.ring_buffer import ThreadSafeRingBuffer
from hardcoded_improv.utils import rms_dbfs

logger = logging.getLogger(__name__)


class LiveAudioInput:
    """Real-time mono audio capture using a callback stream."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.ring_buffer = ThreadSafeRingBuffer(
            capacity_samples=cfg.sample_rate * cfg.ring_buffer_seconds,
            channels=cfg.channels,
        )
        self._last_block_rms = 0.0
        self._level_lock = Lock()
        self._stream: sd.InputStream | None = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            logger.warning("Input stream status: %s", status)

        block = np.asarray(indata, dtype=np.float32)
        if block.ndim == 2 and block.shape[1] > 1:
            block = block[:, :1]
        elif block.ndim == 1:
            block = block[:, None]

        self.ring_buffer.write(block)

        rms = float(np.sqrt(np.mean(np.square(block)) + 1e-12))
        with self._level_lock:
            self._last_block_rms = rms

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            blocksize=self.cfg.blocksize,
            device=self.cfg.input_device,
            channels=self.cfg.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info(
            "Audio input started (sr=%s, blocksize=%s, device=%s)",
            self.cfg.sample_rate,
            self.cfg.blocksize,
            self.cfg.input_device,
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        logger.info("Audio input stopped")

    def get_last_seconds(self, seconds: float) -> np.ndarray:
        return self.ring_buffer.get_last_seconds(seconds, self.cfg.sample_rate)

    def last_level_dbfs(self) -> float:
        with self._level_lock:
            rms = self._last_block_rms
        if rms <= 0:
            return -120.0
        return 20.0 * np.log10(rms)

    def buffered_level_dbfs(self, window_seconds: float = 0.25) -> float:
        return rms_dbfs(self.get_last_seconds(window_seconds))

    def __enter__(self) -> "LiveAudioInput":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_type, exc_val, exc_tb
        self.stop()


def list_input_devices() -> list[tuple[int, str]]:
    devices = sd.query_devices()
    result: list[tuple[int, str]] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            result.append((idx, str(dev.get("name", f"Device {idx}"))))
    return result
