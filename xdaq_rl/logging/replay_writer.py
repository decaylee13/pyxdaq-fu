from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReplayWriter:
    """Simple NPZ episode logger for offline RL bootstrap."""

    out_dir: Path

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._obs = []
        self._actions = []
        self._rewards = []
        self._dones = []

    def append(self, obs: np.ndarray, action: float, reward: float, done: bool) -> None:
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(float(action))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))

    def flush(self, episode_id: int) -> Path:
        path = self.out_dir / f"episode_{episode_id:06d}.npz"
        np.savez_compressed(
            path,
            obs=np.asarray(self._obs, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.float32),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.bool_),
        )
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        return path
