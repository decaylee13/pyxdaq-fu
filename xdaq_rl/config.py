from dataclasses import dataclass


@dataclass(slots=True)
class HardwareConfig:
    mode: str = "sim"  # 'sim' or 'real'
    rhs: bool = False
    device_index: int = 0
    sample_rate_hz: int = 30000
    num_streams: int = 2
    chunk_samples: int = 128
    stream_kind: str = "continuous"  # 'continuous' for RHD, 'AC'/'DC' for RHS


@dataclass(slots=True)
class EnvConfig:
    window_size: int = 1024
    step_horizon: int = 128
    max_steps: int = 1000
    action_low: float = -1.0
    action_high: float = 1.0
