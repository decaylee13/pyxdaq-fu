from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:
        from . import minigym as gym
        from .minigym import spaces

from xdaq_rl.buffer.ring_buffer import SignalRingBuffer
from xdaq_rl.config import EnvConfig
from xdaq_rl.features.base import FeatureExtractor
from xdaq_rl.runtime.acquisition import AcquisitionService

RewardFn = Callable[[np.ndarray, float], float]


def default_reward(obs: np.ndarray, action: float) -> float:
    _ = obs, action
    return 0.0


class XDAQEnv(gym.Env):
    """Gym-style environment over live XDAQ acquisition."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        acquisition: AcquisitionService,
        ring: SignalRingBuffer,
        feature_extractor: FeatureExtractor,
        cfg: EnvConfig,
        reward_fn: RewardFn = default_reward,
    ):
        super().__init__()
        self.acquisition = acquisition
        self.ring = ring
        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.reward_fn = reward_fn

        n_channels = ring.num_channels
        obs_dim = feature_extractor.output_dim(n_channels)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([cfg.action_low], dtype=np.float32),
            high=np.array([cfg.action_high], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count = 0

    def _wait_for_window(self, timeout_s: float = 2.0) -> np.ndarray:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            err = self.acquisition.poll_error()
            if err is not None:
                raise RuntimeError(f"Acquisition error: {err}")

            batch = self.ring.latest(self.cfg.window_size)
            if batch is not None:
                return batch.signal
            time.sleep(0.002)

        raise TimeoutError("Timed out waiting for enough samples in ring buffer")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        _ = options

        if not self.acquisition.running:
            self.acquisition.start()

        self._step_count = 0
        window = self._wait_for_window(timeout_s=5.0)
        obs = self.feature_extractor.compute(window, self.acquisition.backend.sample_rate_hz)
        info = {
            "sample_rate_hz": self.acquisition.backend.sample_rate_hz,
            "num_channels": self.ring.num_channels,
        }
        return obs, info

    def step(self, action: np.ndarray):
        action_val = float(np.asarray(action).reshape(-1)[0])
        self.acquisition.backend.apply_action(action_val)

        latest_before = self.ring.latest_sample_index()
        if latest_before is None:
            latest_before = 0

        target = latest_before + self.cfg.step_horizon
        self.acquisition.wait_until_sample_index(target_index=target, timeout_s=1.0)

        window = self._wait_for_window(timeout_s=1.0)
        obs = self.feature_extractor.compute(window, self.acquisition.backend.sample_rate_hz)
        reward = float(self.reward_fn(obs, action_val))

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.cfg.max_steps
        info = {
            "step": self._step_count,
            "latest_sample_index": self.ring.latest_sample_index(),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self.acquisition.stop()
