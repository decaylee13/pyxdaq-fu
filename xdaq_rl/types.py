from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class SampleBatch:
    """Canonical batch payload passed from acquisition to buffers/features."""

    signal: np.ndarray  # shape: [n_samples, n_channels]
    sample_index: np.ndarray  # shape: [n_samples]
    timestamp_s: Optional[np.ndarray] = None
