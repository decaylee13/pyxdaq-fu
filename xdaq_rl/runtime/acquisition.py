from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

from xdaq_rl.buffer.ring_buffer import SignalRingBuffer
from xdaq_rl.hardware.base import HardwareBackend
from xdaq_rl.types import SampleBatch


class AcquisitionService:
    """
    Owns backend streaming lifecycle and pushes batches into a ring buffer.

    The backend callback path should stay minimal for low latency:
    - append batch to ring buffer
    - publish latest sample index to a small queue for downstream sync
    """

    def __init__(self, backend: HardwareBackend, ring: SignalRingBuffer):
        self.backend = backend
        self.ring = ring
        self._errors: "queue.Queue[str]" = queue.Queue(maxsize=256)
        self._latest_index_q: "queue.Queue[int]" = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self.backend.connect()
            self.backend.start_stream(self._on_batch, self._on_error)
            self._running = True

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self.backend.stop_stream()
            self.backend.disconnect()
            self._running = False

    def _on_batch(self, batch: SampleBatch) -> None:
        self.ring.append(batch)
        latest = int(batch.sample_index[-1])
        try:
            self._latest_index_q.put_nowait(latest)
        except queue.Full:
            try:
                _ = self._latest_index_q.get_nowait()
            except queue.Empty:
                pass
            self._latest_index_q.put_nowait(latest)

    def _on_error(self, err: str) -> None:
        try:
            self._errors.put_nowait(err)
        except queue.Full:
            pass

    def poll_error(self) -> Optional[str]:
        try:
            return self._errors.get_nowait()
        except queue.Empty:
            return None

    def wait_until_sample_index(self, target_index: int, timeout_s: float) -> bool:
        """Wait until the producer reports reaching target sample index."""
        if target_index < 0:
            return True

        from time import monotonic

        t0 = monotonic()
        while monotonic() - t0 < timeout_s:
            if self.ring.latest_sample_index() is not None and self.ring.latest_sample_index() >= target_index:
                return True
            try:
                latest = self._latest_index_q.get(timeout=0.01)
                if latest >= target_index:
                    return True
            except queue.Empty:
                pass
        return False
