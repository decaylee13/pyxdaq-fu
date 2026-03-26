from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from pyxdaq.datablock import Samples
from pyxdaq.simulated_xdaq import SimulatedXDAQ
from xdaq_rl.config import HardwareConfig
from xdaq_rl.hardware.base import HardwareBackend
from xdaq_rl.types import SampleBatch


class SimulatedBackend(HardwareBackend):
    """Adapter over SimulatedXDAQ with the same backend contract as real hardware."""

    def __init__(self, cfg: HardwareConfig):
        self._cfg = cfg
        self.rhs = cfg.rhs
        self._xdaq = SimulatedXDAQ(
            rhs=cfg.rhs,
            num_streams=cfg.num_streams,
            sample_rate=cfg.sample_rate_hz,
        )
        self._stream_cm = None

    @property
    def sample_rate_hz(self) -> int:
        return int(self._xdaq.sampleRate.rate)

    @property
    def num_channels(self) -> int:
        per_stream = 16 if self.rhs else 32
        return self._cfg.num_streams * per_stream

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def _samples_to_batch(self, samples: Samples) -> SampleBatch:
        if self.rhs:
            if self._cfg.stream_kind.upper() == "DC":
                signal = samples.amp[..., 0]
            else:
                signal = samples.amp[..., 1]
        else:
            signal = samples.amp

        signal = signal.reshape(samples.n, -1).astype(np.float32)
        timestamp_s = samples.sample_index.astype(np.float64) / float(self.sample_rate_hz)
        return SampleBatch(signal=signal, sample_index=samples.sample_index, timestamp_s=timestamp_s)

    def start_stream(
        self,
        on_batch: Callable[[SampleBatch], None],
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        def _on_samples(samples: Samples) -> None:
            on_batch(self._samples_to_batch(samples))

        self._stream_cm = self._xdaq.start_receiving_samples(
            callbacks=[_on_samples],
            on_error=on_error,
            chunk_samples=self._cfg.chunk_samples,
        )
        self._stream_cm.__enter__()
        self._xdaq.start(continuous=True)

    def stop_stream(self) -> None:
        self._xdaq.stop(wait=True)
        if self._stream_cm is not None:
            self._stream_cm.__exit__(None, None, None)
            self._stream_cm = None

    def apply_action(self, action: float) -> None:
        # Example action mapping: convert sign to a TTL bit for quick loop testing.
        self._xdaq.setTTLout(0, action > 0)
