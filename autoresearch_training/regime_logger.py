"""
regime_logger.py

Persistent logging across sessions, days, and regime evaluations.

Creates three append-only TSV files:
    regime_log.tsv   — one row per evaluated regime (summary level)
    session_log.tsv  — one row per individual session
    control_log.tsv  — one row per control session

Because experiments span multiple sessions and potentially multiple days,
this logger is more complex than autoresearch/results_logger.py.  The
session_log enables reconstruction of the full learning trajectory for
any regime after the fact.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from autoresearch_training.session_runner import SessionResult
from autoresearch_training.stimulation import StimRegime
from autoresearch_training.learning import LearningCurve


# ---------------------------------------------------------------------------
# Column schemas (defined once, used for headers and row writes)
# ---------------------------------------------------------------------------

_REGIME_COLS = [
    "regime_id", "reward_amp", "penalty_amp", "reward_dur", "penalty_dur",
    "reward_pulses", "penalty_pulses", "slope", "drift_corrected_slope",
    "delta", "n_sessions", "status", "description",
]

_SESSION_COLS = [
    "timestamp", "regime_id", "session_index", "mean_rally_length",
    "hit_rate", "total_rallies", "peak_firing_hz", "stim_enabled",
]

_CONTROL_COLS = [
    "timestamp", "session_index", "mean_rally_length",
    "hit_rate", "total_rallies", "peak_firing_hz",
]


class RegimeLogger:
    """
    Appends experiment data to persistent TSV files in log_dir.

    Parameters
    ----------
    log_dir : Directory where TSV files are written.
              Defaults to autoresearch_training/logs/ relative to the project root.
    """

    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._regime_path  = self._log_dir / "regime_log.tsv"
        self._session_path = self._log_dir / "session_log.tsv"
        self._control_path = self._log_dir / "control_log.tsv"
        self._summary_path = self._log_dir / "summary.md"

        self._ensure_headers()

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_session(self, result: SessionResult, regime: StimRegime) -> None:
        """Append one session row to session_log.tsv."""
        row = {
            "timestamp":          result.timestamp,
            "regime_id":          result.regime_id,
            "session_index":      result.session_index,
            "mean_rally_length":  f"{result.mean_rally_length:.4f}",
            "hit_rate":           f"{result.hit_rate:.4f}",
            "total_rallies":      result.total_rallies,
            "peak_firing_hz":     f"{result.peak_firing_hz:.4f}",
            "stim_enabled":       str(result.stim_enabled).lower(),
        }
        self._append_row(self._session_path, _SESSION_COLS, row)

    def log_control_session(self, result: SessionResult) -> None:
        """Append one control session row to control_log.tsv."""
        row = {
            "timestamp":         result.timestamp,
            "session_index":     result.session_index,
            "mean_rally_length": f"{result.mean_rally_length:.4f}",
            "hit_rate":          f"{result.hit_rate:.4f}",
            "total_rallies":     result.total_rallies,
            "peak_firing_hz":    f"{result.peak_firing_hz:.4f}",
        }
        self._append_row(self._control_path, _CONTROL_COLS, row)

    def log_regime_result(
        self,
        regime: StimRegime,
        curve: LearningCurve,
        drift_corrected_curve: Optional[LearningCurve],
        status: str,
        description: str,
    ) -> None:
        """
        Append one regime summary row to regime_log.tsv.

        Parameters
        ----------
        status : One of "keep" | "discard" | "inconclusive" | "crash".
        """
        dc_slope = drift_corrected_curve.slope if drift_corrected_curve is not None else ""
        row = {
            "regime_id":             regime.regime_id,
            "reward_amp":            f"{regime.reward_amplitude_uv:.1f}",
            "penalty_amp":           f"{regime.penalty_amplitude_uv:.1f}",
            "reward_dur":            f"{regime.reward_duration_ms:.3f}",
            "penalty_dur":           f"{regime.penalty_duration_ms:.3f}",
            "reward_pulses":         regime.reward_n_pulses,
            "penalty_pulses":        regime.penalty_n_pulses,
            "slope":                 f"{curve.slope:.5f}",
            "drift_corrected_slope": f"{dc_slope:.5f}" if dc_slope != "" else "",
            "delta":                 f"{curve.delta:.4f}",
            "n_sessions":            curve.n_sessions,
            "status":                status,
            "description":           description.replace("\t", " "),
        }
        self._append_row(self._regime_path, _REGIME_COLS, row)

    def get_best_regime(self) -> Optional[tuple[str, float]]:
        """
        Return (regime_id, drift_corrected_slope) of the kept regime with the
        highest drift-corrected slope.  Returns None if no kept regimes exist.
        """
        rows = self.get_all_regimes()
        kept = [r for r in rows if r.get("status") == "keep"]
        if not kept:
            return None
        best = max(kept, key=lambda r: _safe_float(r.get("drift_corrected_slope", "")))
        return best["regime_id"], _safe_float(best.get("drift_corrected_slope", "0"))

    def get_all_regimes(self) -> list[dict]:
        """Return all regime rows as a list of dicts."""
        return self._read_tsv(self._regime_path, _REGIME_COLS)

    def write_summary(self, text: str) -> None:
        """Append a comment block to summary.md."""
        ts = datetime.now(timezone.utc).isoformat()
        with open(self._summary_path, "a") as f:
            f.write(f"\n## Summary — {ts}\n\n{text}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_headers(self) -> None:
        for path, cols in [
            (self._regime_path,  _REGIME_COLS),
            (self._session_path, _SESSION_COLS),
            (self._control_path, _CONTROL_COLS),
        ]:
            if not path.exists():
                path.write_text("\t".join(cols) + "\n")

    def _append_row(self, path: Path, cols: list[str], row: dict) -> None:
        line = "\t".join(str(row.get(c, "")) for c in cols) + "\n"
        with open(path, "a") as f:
            f.write(line)

    def _read_tsv(self, path: Path, cols: list[str]) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            return []
        # Skip header and comment lines
        for line in lines[1:]:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            row = {c: (parts[i] if i < len(parts) else "") for i, c in enumerate(cols)}
            rows.append(row)
        return rows


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("-inf")
