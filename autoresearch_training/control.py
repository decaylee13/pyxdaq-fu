"""
control.py

Control condition management for autoresearch_training/.

The control condition (stim disabled) is the mechanism that separates genuine
learning from biological drift.  ControlManager schedules and runs no-stim
sessions at regular intervals, accumulates their results into a LearningCurve,
and exposes the drift slope for use by subtract_drift() in learning.py.

Drift correction is only valid once at least min_control_sessions have been
collected.  Before that threshold, get_drift_curve() and get_drift_rate() return
None and the researcher should flag decisions as "inconclusive".
"""

from __future__ import annotations

import sys
from pathlib import Path

from autoresearch_training.session_runner import SessionResult, run_session
from autoresearch_training.stimulation import StimRegime, disable_stim
from autoresearch_training.learning import LearningCurve, compute_learning_curve


class ControlManager:
    """
    Manages the no-stimulation control condition.

    Parameters
    ----------
    control_interval     : Run a control session every N total sessions.
    session_seconds      : Duration of each control session in seconds.
    min_control_sessions : Minimum number of control sessions before drift
                           correction is considered valid.  Default 3.
    """

    def __init__(
        self,
        control_interval: int,
        session_seconds: int,
        min_control_sessions: int = 3,
        mode: str = "rhx",
        config_path: str | None = None,
    ) -> None:
        if control_interval < 1:
            raise ValueError("control_interval must be >= 1")
        if session_seconds < 1:
            raise ValueError("session_seconds must be >= 1")

        self._interval             = control_interval
        self._session_seconds      = session_seconds
        self._min_sessions         = min_control_sessions
        self._mode                 = mode
        self._config_path          = config_path

        self._control_results: list[SessionResult] = []
        self._control_index: int = 0   # monotonically increasing control session counter

    def should_run_control(self, total_sessions_run: int) -> bool:
        """Return True if a control session is due after total_sessions_run sessions."""
        if total_sessions_run <= 0:
            return False
        return (total_sessions_run % self._interval) == 0

    def run_control_session(
        self,
        session_index: int | None = None,
        current_regime: StimRegime | None = None,
    ) -> SessionResult:
        """
        Run a session with stimulation disabled.

        Parameters
        ----------
        session_index    : Session index to embed in the result.  Defaults to
                           the internal control session counter.
        current_regime   : Active regime to derive pulse parameters from (for
                           logging consistency).  If None, a minimal placeholder
                           regime is constructed.

        Returns
        -------
        SessionResult with stim_enabled=False.
        """
        if session_index is None:
            session_index = self._control_index

        if current_regime is None:
            # Minimal control regime — parameters do not matter since stim is disabled
            current_regime = StimRegime(
                reward_amplitude_uv=200.0,
                reward_duration_ms=0.2,
                reward_frequency_hz=10.0,
                reward_n_pulses=1,
                reward_isi_ms=10.0,
                penalty_amplitude_uv=400.0,
                penalty_duration_ms=0.4,
                penalty_frequency_hz=10.0,
                penalty_n_pulses=1,
                penalty_isi_ms=10.0,
                reward_penalty_ratio=0.5,
                stim_enabled=False,
            )

        control_regime = disable_stim(current_regime)

        result = run_session(
            regime=control_regime,
            session_seconds=self._session_seconds,
            session_index=session_index,
            mode=self._mode,
            config_path=self._config_path,
        )

        self._control_results.append(result)
        self._control_index += 1

        print(
            f"[control] Control session {self._control_index} complete — "
            f"mean_rally={result.mean_rally_length:.3f}, "
            f"peak_hz={result.peak_firing_hz:.1f}",
            file=sys.stderr,
        )
        return result

    def get_drift_curve(self) -> LearningCurve | None:
        """
        Return a LearningCurve built from all control sessions collected so far.

        Returns None if fewer than min_control_sessions have been run — drift
        correction is not valid until this threshold is reached.
        """
        if len(self._control_results) < self._min_sessions:
            return None
        # Re-index control results sequentially for regression (0, 1, 2, ...)
        reindexed = [
            SessionResult(
                mean_rally_length=r.mean_rally_length,
                hit_rate=r.hit_rate,
                total_rallies=r.total_rallies,
                peak_firing_hz=r.peak_firing_hz,
                session_seconds=r.session_seconds,
                regime_id="control",
                session_index=i,
                timestamp=r.timestamp,
                stim_enabled=False,
                crashed=r.crashed,
            )
            for i, r in enumerate(self._control_results)
        ]
        return compute_learning_curve(reindexed)

    def get_drift_rate(self) -> float | None:
        """
        Return the slope of control sessions (performance change per session without stim).

        Positive: culture spontaneously improving.
        Negative: culture declining.
        None: not enough control sessions yet.
        """
        curve = self.get_drift_curve()
        if curve is None:
            return None
        return curve.slope

    @property
    def n_control_sessions(self) -> int:
        return len(self._control_results)

    @property
    def all_control_results(self) -> list[SessionResult]:
        return list(self._control_results)
