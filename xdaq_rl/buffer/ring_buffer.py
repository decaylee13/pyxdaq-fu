from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from xdaq_rl.types import SampleBatch


class SignalRingBuffer:
    """Thread-safe ring buffer for multi-channel neural signals."""

    def __init__(self, capacity_samples: int, num_channels: int):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")

        self.capacity = int(capacity_samples)
        self.num_channels = int(num_channels)

        self._signal = np.zeros((self.capacity, self.num_channels), dtype=np.float32)
        self._sample_index = np.zeros((self.capacity,), dtype=np.uint64)
        self._timestamp_s = np.zeros((self.capacity,), dtype=np.float64)

        self._write_ptr = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def clear(self) -> None:
        with self._lock:
            self._write_ptr = 0
            self._size = 0

    def append(self, batch: SampleBatch) -> None:
        n = int(batch.signal.shape[0])
        if n == 0:
            return

        if batch.signal.shape[1] != self.num_channels:
            raise ValueError(
                f"Channel mismatch. got={batch.signal.shape[1]}, expected={self.num_channels}"
            )

        idx = batch.sample_index.astype(np.uint64, copy=False)
        if batch.timestamp_s is None:
            ts = np.zeros((n,), dtype=np.float64)
        else:
            ts = batch.timestamp_s.astype(np.float64, copy=False)

        with self._lock:
            if n >= self.capacity:
                self._signal[:, :] = batch.signal[-self.capacity :]
                self._sample_index[:] = idx[-self.capacity :]
                self._timestamp_s[:] = ts[-self.capacity :]
                self._write_ptr = 0
                self._size = self.capacity
                return

            end = self._write_ptr + n
            if end <= self.capacity:
                self._signal[self._write_ptr:end] = batch.signal
                self._sample_index[self._write_ptr:end] = idx
                self._timestamp_s[self._write_ptr:end] = ts
            else:
                k = self.capacity - self._write_ptr
                self._signal[self._write_ptr:] = batch.signal[:k]
                self._signal[: n - k] = batch.signal[k:]
                self._sample_index[self._write_ptr:] = idx[:k]
                self._sample_index[: n - k] = idx[k:]
                self._timestamp_s[self._write_ptr:] = ts[:k]
                self._timestamp_s[: n - k] = ts[k:]

            self._write_ptr = (self._write_ptr + n) % self.capacity
            self._size = min(self.capacity, self._size + n)

    def latest(self, n_samples: int) -> Optional[SampleBatch]:
        with self._lock:
            if n_samples <= 0:
                raise ValueError("n_samples must be positive")
            if self._size < n_samples:
                return None

            start = (self._write_ptr - n_samples) % self.capacity
            if start < self._write_ptr:
                sig = self._signal[start:self._write_ptr].copy()
                idx = self._sample_index[start:self._write_ptr].copy()
                ts = self._timestamp_s[start:self._write_ptr].copy()
            else:
                sig = np.concatenate(
                    [self._signal[start:].copy(), self._signal[:self._write_ptr].copy()], axis=0
                )
                idx = np.concatenate(
                    [self._sample_index[start:].copy(), self._sample_index[:self._write_ptr].copy()]
                )
                ts = np.concatenate(
                    [self._timestamp_s[start:].copy(), self._timestamp_s[:self._write_ptr].copy()]
                )

        return SampleBatch(signal=sig, sample_index=idx, timestamp_s=ts)

    def latest_sample_index(self) -> Optional[int]:
        with self._lock:
            if self._size == 0:
                return None
            pos = (self._write_ptr - 1) % self.capacity
            return int(self._sample_index[pos])
