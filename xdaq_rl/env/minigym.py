from __future__ import annotations

import numpy as np


class Env:
    metadata = {}

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        return None

    def step(self, action):
        raise NotImplementedError

    def close(self):
        return


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        low_arr = np.asarray(low, dtype=dtype)
        high_arr = np.asarray(high, dtype=dtype)
        if shape is None:
            shape = tuple(low_arr.shape)
            if not shape:
                shape = tuple(high_arr.shape)
        if not shape:
            raise ValueError("shape could not be inferred for Box")

        self.low = np.broadcast_to(low_arr, shape).astype(dtype, copy=False)
        self.high = np.broadcast_to(high_arr, shape).astype(dtype, copy=False)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)


class spaces:
    Box = Box
