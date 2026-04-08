"""
results_logger.py

Manages the results.tsv experiment log: appending rows, retrieving best
results, loading recent history, detecting monotonic performance decline,
and writing human-readable summary comments into the file.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Optional

from autoresearch.metrics import SessionResult


# TSV column order
_COLUMNS = ["commit", "mean_rally_length", "hit_rate", "status", "description"]


class ResultsLogger:
    """
    Append-only TSV log of every experiment.

    File format:
        commit<TAB>mean_rally_length<TAB>hit_rate<TAB>status<TAB>description

    Lines beginning with ``#`` are comment/summary blocks and are ignored
    when parsing.
    """

    def __init__(self, tsv_path: str) -> None:
        self._path = Path(tsv_path)
        self._ensure_header()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(
        self,
        commit_hash: str,
        result: SessionResult,
        status: str,
        description: str,
    ) -> None:
        """
        Append one experiment row.

        Parameters
        ----------
        commit_hash : str
            Short git hash identifying this config state.
        result : SessionResult
            The measured performance metrics.
        status : str
            "keep" | "discard" | "crash"
        description : str
            Human-readable description from describe_perturbation().
        """
        row = {
            "commit":            commit_hash,
            "mean_rally_length": f"{result.mean_rally_length:.4f}",
            "hit_rate":          f"{result.hit_rate:.4f}",
            "status":            status,
            "description":       description,
        }
        with self._path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS, delimiter="\t")
            writer.writerow(row)

    def write_summary(self, summary_text: str) -> None:
        """Append a comment block to the TSV (lines prefixed with #)."""
        lines = "\n".join(f"# {line}" for line in summary_text.splitlines())
        with self._path.open("a") as fh:
            fh.write(f"\n{lines}\n")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_best(self) -> Optional[tuple[str, SessionResult]]:
        """
        Return (commit_hash, SessionResult) of the best *kept* experiment
        ranked by mean_rally_length.  Returns None if no kept rows exist.
        """
        best_row: Optional[dict] = None
        best_score: float = -1.0

        for row in self._iter_data_rows():
            if row["status"] != "keep":
                continue
            score = float(row["mean_rally_length"])
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is None:
            return None

        return (
            best_row["commit"],
            self._row_to_result(best_row),
        )

    def get_recent(self, n: int) -> list[dict]:
        """Return the last N data rows as plain dicts."""
        rows = list(self._iter_data_rows())
        return rows[-n:]

    def detect_monotonic_decline(self, window: int = 5) -> bool:
        """
        Return True if mean_rally_length has declined monotonically across
        the last `window` *kept* experiments (a sign of culture degradation
        or a systematically wrong search direction).
        """
        kept = [
            float(r["mean_rally_length"])
            for r in self._iter_data_rows()
            if r["status"] == "keep"
        ]
        if len(kept) < window:
            return False
        recent = kept[-window:]
        return all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_header(self) -> None:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS, delimiter="\t")
                writer.writeheader()

    def _iter_data_rows(self):
        """Yield parsed data rows (skip header and # comment lines)."""
        if not self._path.exists():
            return
        with self._path.open(newline="") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # skip TSV header
                if stripped.startswith("commit"):
                    continue
                row_io = io.StringIO(stripped)
                reader = csv.DictReader(row_io, fieldnames=_COLUMNS, delimiter="\t")
                for row in reader:
                    yield row

    @staticmethod
    def _row_to_result(row: dict) -> SessionResult:
        from autoresearch.metrics import SessionResult
        from datetime import datetime, timezone
        return SessionResult(
            mean_rally_length=float(row["mean_rally_length"]),
            hit_rate=float(row["hit_rate"]),
            total_rallies=0,
            peak_firing_hz=0.0,
            session_seconds=0.0,
            config_hash="",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
