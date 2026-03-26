from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from xdaq_rl.types import SampleBatch


class HardwareBackend(ABC):
    """Abstract backend contract used by acquisition/runtime layers."""

    rhs: bool

    @property
    @abstractmethod
    def sample_rate_hz(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_channels(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_stream(
        self,
        on_batch: Callable[[SampleBatch], None],
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop_stream(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action: float) -> None:
        """Apply a control action. Keep as a no-op unless configured."""
        raise NotImplementedError
