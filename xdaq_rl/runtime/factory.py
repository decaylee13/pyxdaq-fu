from __future__ import annotations

from xdaq_rl.buffer.ring_buffer import SignalRingBuffer
from xdaq_rl.config import HardwareConfig
from xdaq_rl.hardware.base import HardwareBackend
from xdaq_rl.hardware.pyxdaq_backend import PyXDAQBackend
from xdaq_rl.hardware.simulated_backend import SimulatedBackend


def create_backend(cfg: HardwareConfig) -> HardwareBackend:
    mode = cfg.mode.lower()
    if mode == "real":
        return PyXDAQBackend(cfg)
    if mode == "sim":
        return SimulatedBackend(cfg)
    raise ValueError(f"Unsupported mode: {cfg.mode}")


def create_ring(cfg: HardwareConfig) -> SignalRingBuffer:
    per_stream = 16 if cfg.rhs else 32
    num_channels = cfg.num_streams * per_stream
    capacity = cfg.sample_rate_hz * 10
    return SignalRingBuffer(capacity_samples=capacity, num_channels=num_channels)
