from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FeatureExtractor(ABC):
    """Transforms a signal window [T, C] into an observation vector."""

    @abstractmethod
    def output_dim(self, num_channels: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def compute(self, window: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        raise NotImplementedError
