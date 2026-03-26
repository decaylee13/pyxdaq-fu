#!/usr/bin/env python3
import threading
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from pyxdaq.datablock import Samples
from pyxdaq.constants import SampleRate

@dataclass
class _SR:
    rate: int

class _SampleStreamSession:
    def __init__(self, sim, callbacks, on_error, chunk_samples):
        self.sim = sim
        self.callbacks = callbacks
        self.on_error = on_error
        self.chunk_samples = chunk_samples

    def __enter__(self):
        self.sim._start_stream_thread(self.callbacks, self.on_error, self.chunk_samples)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sim._stop_stream_thread()

class SimulatedXDAQ:
    def __init__(self, rhs=False, num_streams=2, sample_rate=30000):
        self.rhs = rhs
        self._num_streams = num_streams
        self.sampleRate = _SR(sample_rate)
        self._running = False
        self._t = 0

        self._stream_thread = None
        self._stop_event = threading.Event()
        self._ttl_out_mask = 0
        self._stim_cmd_mode = False
        self._stim_state = {}

    @property
    def numDataStream(self):
        return self._num_streams

    def start(self, *, continuous: bool = None):
        self._running = True

    def stop(self, *, wait: bool = False):
        self._running = False

    def _stream_loop(self, callbacks, on_error, chunk_samples):
        period_s = chunk_samples / self.sampleRate.rate
        callback_arity = [len(inspect.signature(cb).parameters) for cb in callbacks]

        while not self._stop_event.is_set():
            if not self._running:
                self._stop_event.wait(0.01)
                continue

            try:
                samples = self._make_samples(chunk_samples)
                for cb, arity in zip(callbacks, callback_arity):
                    if arity == 1:
                        cb(samples)
                    elif arity == 2:
                        cb(samples, None)
                    else:
                        raise ValueError("Callback must accept 1 or 2 args")
            except Exception as e:
                if on_error is not None:
                    on_error(str(e))

            self._stop_event.wait(period_s)

    def _start_stream_thread(self, callbacks, on_error, chunk_samples):
        if self._stream_thread and self._stream_thread.is_alive():
            raise RuntimeError("Stream already running")
        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callbacks, on_error, chunk_samples),
            daemon=True,
        )
        self._stream_thread.start()

    def _stop_stream_thread(self):
        self._stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
        self._stream_thread = None

    def start_receiving_samples(
        self,
        callbacks: list[Union[Callable[[Samples], None], Callable[[Samples | None, str | None], None]]],
        on_error: Optional[Callable[[str], None]],
        chunk_samples: int = 128,
    ):
        if not callbacks:
            raise ValueError("At least one callback must be provided")
        for i, cb in enumerate(callbacks):
            if not callable(cb):
                raise ValueError(f"Callback at index {i} is not callable")
            arity = len(inspect.signature(cb).parameters)
            if arity not in (1, 2):
                raise ValueError(f"Callback at index {i} has invalid signature")
        return _SampleStreamSession(self, callbacks, on_error, chunk_samples)

    def getSampleRate(self) -> int:
        return self.sampleRate.rate

    def changeSampleRate(
        self,
        sampleRate: Union[SampleRate, int],
        fastSettle: bool = False,
        update_stim: bool = False,
        upper_bandwidth: float = 7500,
        lower_bandwidth: float = 1,
    ):
        if isinstance(sampleRate, SampleRate):
            self.sampleRate = _SR(sampleRate.rate)
        else:
            self.sampleRate = _SR(int(sampleRate))

    def setTTLout(self, channel: Union[int, str], enable: Union[bool, int]):
        if isinstance(channel, str) and channel == "all":
            if isinstance(enable, bool):
                self._ttl_out_mask = 0xFFFFFFFF if enable else 0
            else:
                self._ttl_out_mask = int(enable) & 0xFFFFFFFF
            return

        ch = int(channel)
        if ch < 0 or ch > 31:
            raise ValueError("channel out of range")
        if bool(enable):
            self._ttl_out_mask |= 1 << ch
        else:
            self._ttl_out_mask &= ~(1 << ch)

    def setStimCmdMode(self, enabled: bool):
        self._stim_cmd_mode = bool(enabled)

    def set_stim(self, **kwargs):
        stream = int(kwargs["stream"])
        channel = int(kwargs["channel"])
        self._stim_state[(stream, channel)] = dict(kwargs)

    def _make_samples(self, n=128):
        idx = np.arange(self._t, self._t + n, dtype=np.uint32)
        t = idx.astype(np.float32) / self.sampleRate.rate
        base = 32768 + 400*np.sin(2*np.pi*20*t)

        if self.rhs:
            amp = np.zeros((n, self._num_streams, 16, 2), dtype=np.uint16)
            ac = np.clip(base[:, None, None] + np.random.randn(n, self._num_streams, 16)*40, 0, 65535)
            dc = np.clip(512 + np.random.randn(n, self._num_streams, 16)*3, 0, 1023)
            amp[..., 1] = ac.astype(np.uint16)
            amp[..., 0] = dc.astype(np.uint16)
            dac = np.zeros((n, 8), dtype=np.uint16)
            stim = np.zeros((n, 4, self._num_streams), dtype=np.uint16)
            aux = np.zeros((n, 4, self._num_streams, 2), dtype=np.uint16)
        else:
            amp = np.clip(base[:, None, None] + np.random.randn(n, self._num_streams, 32)*40, 0, 65535).astype(np.uint16)
            dac = None
            stim = None
            aux = np.zeros((n, 3, self._num_streams), dtype=np.uint16)

        self._t += n
        return Samples(
            sample_index=idx,
            aux=aux,
            amp=amp,
            timestamp=None,
            adc=np.zeros((n, 8), dtype=np.uint16),
            ttlin=np.zeros((n, 1), dtype=np.uint32),
            ttlout=np.full((n, 1), self._ttl_out_mask, dtype=np.uint32),
            dac=dac,
            stim=stim,
            n=n,
        )
