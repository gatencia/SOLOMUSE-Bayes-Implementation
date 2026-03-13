from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass
class RingBufferStats:
    capacity_samples: int
    current_size: int
    write_index: int


class ThreadSafeRingBuffer:
    """Thread-safe mono/stereo ring buffer for float32 audio samples."""

    def __init__(self, capacity_samples: int, channels: int = 1) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")

        self.capacity_samples = int(capacity_samples)
        self.channels = int(channels)
        self._buffer = np.zeros((self.capacity_samples, self.channels), dtype=np.float32)
        self._write_index = 0
        self._size = 0
        self._lock = Lock()

    def write(self, block: np.ndarray) -> None:
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("block must be 1D or 2D")
        if arr.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {arr.shape[1]}"
            )
        if arr.shape[0] == 0:
            return

        if arr.shape[0] > self.capacity_samples:
            arr = arr[-self.capacity_samples :]

        n = arr.shape[0]
        with self._lock:
            end = self._write_index + n
            if end <= self.capacity_samples:
                self._buffer[self._write_index : end] = arr
            else:
                first_len = self.capacity_samples - self._write_index
                self._buffer[self._write_index :] = arr[:first_len]
                self._buffer[: end % self.capacity_samples] = arr[first_len:]

            self._write_index = end % self.capacity_samples
            self._size = min(self.capacity_samples, self._size + n)

    def get_last(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        with self._lock:
            n = min(int(num_samples), self._size)
            if n == 0:
                return np.zeros((0, self.channels), dtype=np.float32)

            start = (self._write_index - n) % self.capacity_samples
            if start < self._write_index:
                out = self._buffer[start : self._write_index].copy()
            else:
                out = np.concatenate(
                    (self._buffer[start:], self._buffer[: self._write_index]),
                    axis=0,
                )

        return out

    def get_last_seconds(self, seconds: float, sample_rate: int) -> np.ndarray:
        if seconds <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        num_samples = int(round(seconds * sample_rate))
        return self.get_last(num_samples)

    def stats(self) -> RingBufferStats:
        with self._lock:
            return RingBufferStats(
                capacity_samples=self.capacity_samples,
                current_size=self._size,
                write_index=self._write_index,
            )
