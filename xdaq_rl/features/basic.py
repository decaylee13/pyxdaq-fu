from __future__ import annotations

import numpy as np

from xdaq_rl.features.base import FeatureExtractor


class RMSFeatureExtractor(FeatureExtractor):
    """Simple baseline extractor: per-channel RMS and mean absolute value."""

    def output_dim(self, num_channels: int) -> int:
        return num_channels * 2

    def compute(self, window: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        _ = sample_rate_hz
        if window.ndim != 2:
            raise ValueError("window must have shape [time, channels]")

        w = window.astype(np.float32, copy=False)
        rms = np.sqrt(np.mean(np.square(w), axis=0))
        mav = np.mean(np.abs(w), axis=0)
        return np.concatenate([rms, mav], axis=0).astype(np.float32)
