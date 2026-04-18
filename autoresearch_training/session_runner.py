"""
session_runner.py

Runs a single timed training session under a given StimRegime and returns
performance metrics as a SessionResult.

Each session is executed as an isolated subprocess (_training_session_worker)
so that hardware state is cleanly reset between sessions.  The parent process
captures stdout and parses the standardised metrics block.

Stimulator dependency:
    stimulator.py's StimConfig does not support runtime parameter changes —
    amplitude and duration are set at initialisation and cannot be updated
    per-regime.  The _training_session_worker handles this by constructing
    StimConfig from the CLI-passed regime parameters each time it runs.
    No changes to stimulator.py are required for single-pulse regimes.

    For burst stimulation (reward_n_pulses > 1 or penalty_n_pulses > 1),
    the worker uses its own _BurstStimulator implementation.  If the burst
    logic is eventually merged into stimulator.py, the recommended minimal
    interface change is:
        StimConfig.reward_n_pulses: int = 1
        StimConfig.reward_isi_ms:   float = 10.0
        StimConfig.penalty_n_pulses: int = 1
        StimConfig.penalty_isi_ms:   float = 10.0
    with a corresponding burst loop in Stimulator._trigger().

Culture health:
    If peak_firing_hz < MIN_PEAK_FIRING_HZ (5.0 Hz) the result is returned
    with crashed=False but a warning is printed to stderr.  The caller should
    check this threshold and consider stopping the loop if it persists.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from autoresearch_training.stimulation import StimRegime

# Minimum firing rate below which a result is flagged as potentially unreliable
MIN_PEAK_FIRING_HZ: float = 5.0


@dataclass
class SessionResult:
    mean_rally_length: float
    hit_rate: float
    total_rallies: int
    peak_firing_hz: float
    session_seconds: float
    regime_id: str
    session_index: int        # which session number within this regime evaluation
    timestamp: str
    stim_enabled: bool
    crashed: bool = False


def run_session(
    regime: StimRegime,
    session_seconds: int,
    session_index: int = 0,
    mode: str = "rhx",
    threshold: float = 3.0,
    smooth_windows: int = 1,
    host: str = "127.0.0.1",
    command_port: int = 5000,
    spike_port: int = 5002,
    sample_rate_hz: float = 30000.0,
    config_path: str | None = None,
) -> SessionResult:
    """
    Run a single session under the given StimRegime and return its metrics.

    Parameters
    ----------
    regime          : StimRegime to apply for this session.
    session_seconds : Duration of the session in seconds (required, no default).
    session_index   : Index of this session within the current regime evaluation.
    mode            : "rhx" for live hardware, "mock" for testing without hardware.
    config_path     : Path to channel_config.json.  Defaults to project root.
    """
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "channel_config.json")

    cmd = [
        sys.executable, "-m", "autoresearch_training._training_session_worker",
        "--config-path",          config_path,
        "--session-seconds",      str(float(session_seconds)),
        "--mode",                 mode,
        "--threshold",            str(threshold),
        "--smooth-windows",       str(smooth_windows),
        "--host",                 host,
        "--command-port",         str(command_port),
        "--spike-port",           str(spike_port),
        "--sample-rate-hz",       str(sample_rate_hz),
        "--regime-id",            regime.regime_id,
        "--session-index",        str(session_index),
        "--reward-amplitude-uv",  str(regime.reward_amplitude_uv),
        "--reward-duration-ms",   str(regime.reward_duration_ms),
        "--reward-n-pulses",      str(regime.reward_n_pulses),
        "--reward-isi-ms",        str(regime.reward_isi_ms),
        "--penalty-amplitude-uv", str(regime.penalty_amplitude_uv),
        "--penalty-duration-ms",  str(regime.penalty_duration_ms),
        "--penalty-n-pulses",     str(regime.penalty_n_pulses),
        "--penalty-isi-ms",       str(regime.penalty_isi_ms),
    ]

    if not regime.stim_enabled:
        cmd.append("--no-stim")

    timeout_s = session_seconds * 2

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        stdout = proc.stdout.decode(errors="replace")
        stderr = proc.stderr.decode(errors="replace")
    except subprocess.TimeoutExpired:
        print(
            f"[session_runner] Session {session_index} (regime {regime.regime_id}) "
            f"timed out after {timeout_s}s",
            file=sys.stderr,
        )
        return _crashed_result(regime, session_index)
    except Exception as exc:
        print(f"[session_runner] Subprocess error: {exc}", file=sys.stderr)
        return _crashed_result(regime, session_index)

    result = _parse_training_log(stdout, regime, session_index)

    if result is None:
        if stderr:
            print(f"[session_runner] Worker stderr:\n{stderr}", file=sys.stderr)
        return _crashed_result(regime, session_index)

    if result.peak_firing_hz < MIN_PEAK_FIRING_HZ:
        print(
            f"[session_runner] WARNING: peak_firing_hz={result.peak_firing_hz:.2f} Hz "
            f"is below threshold {MIN_PEAK_FIRING_HZ} Hz — culture may be unhealthy "
            f"(session {session_index}, regime {regime.regime_id})",
            file=sys.stderr,
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_training_log(
    log_text: str,
    regime: StimRegime,
    session_index: int,
) -> SessionResult | None:
    """
    Parse the standardised metrics block from worker stdout.

    Uses autoresearch.metrics.parse_log() for the core fields (mean_rally_length,
    hit_rate, total_rallies, peak_firing_hz, session_seconds) then overlays the
    training-specific fields (regime_id, session_index, stim_enabled).
    """
    try:
        from autoresearch.metrics import parse_log as _base_parse
        base = _base_parse(log_text)
    except ImportError:
        base = None

    if base is None:
        # autoresearch not importable or block absent
        base = _parse_metrics_block(log_text)
        if base is None:
            return None

    # Extract training-specific fields from the raw block
    stim_enabled = regime.stim_enabled   # default to regime value
    m = re.search(r"stim_enabled\s*:\s*(\S+)", log_text)
    if m:
        stim_enabled = m.group(1).lower() in ("true", "1", "yes")

    return SessionResult(
        mean_rally_length=base.mean_rally_length,
        hit_rate=base.hit_rate,
        total_rallies=base.total_rallies,
        peak_firing_hz=base.peak_firing_hz,
        session_seconds=base.session_seconds,
        regime_id=regime.regime_id,
        session_index=session_index,
        timestamp=datetime.now(timezone.utc).isoformat(),
        stim_enabled=stim_enabled,
        crashed=False,
    )


def _parse_metrics_block(log_text: str) -> "object | None":
    """
    Fallback parser used when autoresearch.metrics is unavailable.
    Returns a SimpleNamespace with the same fields as autoresearch.metrics.SessionResult.
    """
    import types

    marker = "---"
    idx = log_text.rfind(marker)
    if idx == -1:
        return None

    block = log_text[idx:]
    data: dict[str, str] = {}
    for line in block.splitlines():
        m = re.match(r"^\s*([\w_]+)\s*:\s*(.+)$", line)
        if m:
            data[m.group(1).strip()] = m.group(2).strip()

    required = {"mean_rally_length", "hit_rate", "total_rallies", "peak_firing_hz", "session_seconds"}
    if not required.issubset(data.keys()):
        return None

    try:
        obj = types.SimpleNamespace(
            mean_rally_length=float(data["mean_rally_length"]),
            hit_rate=float(data["hit_rate"]),
            total_rallies=int(data["total_rallies"]),
            peak_firing_hz=float(data["peak_firing_hz"]),
            session_seconds=float(data["session_seconds"]),
        )
        return obj
    except (ValueError, KeyError):
        return None


def _crashed_result(regime: StimRegime, session_index: int) -> SessionResult:
    return SessionResult(
        mean_rally_length=0.0,
        hit_rate=0.0,
        total_rallies=0,
        peak_firing_hz=0.0,
        session_seconds=0.0,
        regime_id=regime.regime_id,
        session_index=session_index,
        timestamp=datetime.now(timezone.utc).isoformat(),
        stim_enabled=regime.stim_enabled,
        crashed=True,
    )
