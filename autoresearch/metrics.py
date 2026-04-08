"""
metrics.py

Defines the canonical SessionResult dataclass for one session's performance
metrics and provides utilities for parsing the standardised stdout block
emitted by the session worker, averaging repeated sessions, and computing
a short config hash for traceability across experiments.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SessionResult:
    mean_rally_length: float
    hit_rate: float
    total_rallies: int
    peak_firing_hz: float
    session_seconds: float
    config_hash: str
    timestamp: str       # ISO format
    crashed: bool = False


def compute_config_hash(config: dict) -> str:
    """Return a 7-character SHA-256 hash of the config JSON (stable, sorted keys)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:7]


def average_results(results: list[SessionResult]) -> SessionResult:
    """
    Average N SessionResults into one for noise reduction across repeats.
    Crashed sessions are excluded from the average; if all crashed the
    returned result has crashed=True and all metrics at 0.0.
    """
    if not results:
        raise ValueError("Cannot average an empty list of SessionResults")
    if len(results) == 1:
        return results[0]

    non_crashed = [r for r in results if not r.crashed]
    n = len(results)

    if not non_crashed:
        return SessionResult(
            mean_rally_length=0.0,
            hit_rate=0.0,
            total_rallies=0,
            peak_firing_hz=0.0,
            session_seconds=sum(r.session_seconds for r in results) / n,
            config_hash=results[0].config_hash,
            timestamp=results[0].timestamp,
            crashed=True,
        )

    nc = len(non_crashed)
    return SessionResult(
        mean_rally_length=sum(r.mean_rally_length for r in non_crashed) / nc,
        hit_rate=sum(r.hit_rate for r in non_crashed) / nc,
        total_rallies=sum(r.total_rallies for r in non_crashed) // nc,
        peak_firing_hz=max(r.peak_firing_hz for r in non_crashed),
        session_seconds=sum(r.session_seconds for r in non_crashed) / nc,
        config_hash=non_crashed[0].config_hash,
        timestamp=non_crashed[0].timestamp,
        crashed=False,
    )


def parse_log(log_text: str) -> Optional[SessionResult]:
    """
    Parse the standardised metrics block emitted by _session_worker.py.

    Expected block (must appear somewhere in log_text):
        ---
        mean_rally_length: 4.21
        hit_rate:          0.61
        total_rallies:     18
        peak_firing_hz:    32.4
        session_seconds:   90.0
        config_hash:       a3f2b1

    Returns None if the block is absent (crash/timeout detection).
    """
    marker = "---"
    idx = log_text.rfind(marker)   # use last occurrence in case of retries
    if idx == -1:
        return None

    block = log_text[idx:]
    data: dict[str, str] = {}
    for line in block.splitlines():
        m = re.match(r"^\s*([\w_]+)\s*:\s*(.+)$", line)
        if m:
            data[m.group(1).strip()] = m.group(2).strip()

    required = {
        "mean_rally_length", "hit_rate", "total_rallies",
        "peak_firing_hz", "session_seconds", "config_hash",
    }
    if not required.issubset(data.keys()):
        return None

    try:
        return SessionResult(
            mean_rally_length=float(data["mean_rally_length"]),
            hit_rate=float(data["hit_rate"]),
            total_rallies=int(data["total_rallies"]),
            peak_firing_hz=float(data["peak_firing_hz"]),
            session_seconds=float(data["session_seconds"]),
            config_hash=data["config_hash"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except (ValueError, KeyError):
        return None
