from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from pyxdaq.datablock import Samples
from pyxdaq.xdaq import get_XDAQ
from xdaq_rl.config import HardwareConfig
from xdaq_rl.hardware.base import HardwareBackend
from xdaq_rl.types import SampleBatch


class PyXDAQBackend(HardwareBackend):
    """
    Thin wrapper around pyxdaq runtime APIs.

    All pyxdaq-specific behavior is isolated here so env/training code stays backend-agnostic.
    """

    def __init__(self, cfg: HardwareConfig):
        self._cfg = cfg
        self.rhs = cfg.rhs
        self._xdaq = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def sample_rate_hz(self) -> int:
        if self._xdaq is None:
            return self._cfg.sample_rate_hz
        return int(self._xdaq.sampleRate.rate)

    @property
    def num_channels(self) -> int:
        streams = self._cfg.num_streams
        per_stream = 16 if self.rhs else 32
        return streams * per_stream

    def connect(self) -> None:
        self._xdaq = get_XDAQ(rhs=self.rhs, index=self._cfg.device_index)
        # Note: pyxdaq exposes sampleRate as an object with .rate.
        # TODO: call changeSampleRate here if you need non-default rates on hardware.

    def disconnect(self) -> None:
        if self._xdaq is not None:
            try:
                self._xdaq.stop(wait=True)
            except Exception:
                pass
        self._xdaq = None

    def _samples_to_batch(self, samples: Samples) -> SampleBatch:
        if self.rhs:
            if self._cfg.stream_kind.upper() == "DC":
                signal = samples.amp[..., 0]
            else:
                signal = samples.amp[..., 1]
        else:
            signal = samples.amp

        signal = signal.reshape(samples.n, -1).astype(np.float32)

        if samples.timestamp is not None:
            timestamp_s = samples.timestamp.astype(np.float64) / 1_000_000.0
        else:
            timestamp_s = samples.sample_index.astype(np.float64) / float(self.sample_rate_hz)

        return SampleBatch(
            signal=signal,
            sample_index=samples.sample_index.astype(np.uint64, copy=False),
            timestamp_s=timestamp_s,
        )

    def start_stream(
        self,
        on_batch: Callable[[SampleBatch], None],
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._xdaq is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        if self._stream_thread and self._stream_thread.is_alive():
            raise RuntimeError("Stream is already running")

        self._stop_event.clear()

        def _run() -> None:
            def _on_samples(samples: Samples) -> None:
                on_batch(self._samples_to_batch(samples))

            def _on_error(err: str) -> None:
                if on_error is not None:
                    on_error(err)

            with self._xdaq.start_receiving_samples([_on_samples], _on_error):
                self._xdaq.start(continuous=True)
                while not self._stop_event.wait(0.01):
                    pass
                self._xdaq.stop(wait=True)

        self._stream_thread = threading.Thread(target=_run, name="pyxdaq-stream", daemon=True)
        self._stream_thread.start()

    def stop_stream(self) -> None:
        self._stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
        self._stream_thread = None

    def apply_action(self, action: float) -> None:
        # TODO: map action -> TTL/DAC/stim for closed-loop control.
        # Keep no-op while observation/reward pipeline is being validated.
        _ = action
        time.sleep(0.0)
