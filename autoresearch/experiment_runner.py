"""
experiment_runner.py

Launches timed game sessions against live RHX hardware (or a mock controller)
as isolated subprocesses, parses the stdout metrics block, and returns a
SessionResult.  Running each session in its own subprocess ensures that
hardware socket state is fully reset between experiments.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from autoresearch.metrics import SessionResult, average_results, parse_log


# Path to the headless session worker script
_WORKER = Path(__file__).parent / "_session_worker.py"


def run_session(
    config_path: str,
    session_seconds: int = 90,
    n_repeats: int = 1,
    mode: str = "rhx",
    threshold: float = 3.0,
    smooth_windows: int = 1,
    host: str = "127.0.0.1",
    command_port: int = 5000,
    spike_port: int = 5002,
    sample_rate_hz: float = 30000.0,
    no_stim: bool = False,
) -> SessionResult:
    """
    Run one or more timed Pong sessions and return the averaged result.

    Parameters
    ----------
    config_path : str
        Path to channel_config.json to use for this session.
    session_seconds : int
        Duration of each session in seconds.
    n_repeats : int
        Number of sessions to run and average (reduces measurement noise).
    mode : str
        "rhx" for live hardware, "mock" for headless testing.
    threshold, smooth_windows : float, int
        ActionDecoder parameters forwarded to the session worker.
    host, command_port, spike_port, sample_rate_hz
        RHX TCP connection parameters.

    Returns
    -------
    SessionResult
        Averaged result across repeats.  result.crashed=True if all
        sessions failed.
    """
    results: list[SessionResult] = []
    for _ in range(n_repeats):
        result = _run_one(
            config_path=config_path,
            session_seconds=session_seconds,
            mode=mode,
            threshold=threshold,
            smooth_windows=smooth_windows,
            host=host,
            command_port=command_port,
            spike_port=spike_port,
            sample_rate_hz=sample_rate_hz,
            no_stim=no_stim,
        )
        results.append(result)

    return average_results(results)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _run_one(
    config_path: str,
    session_seconds: int,
    mode: str,
    threshold: float,
    smooth_windows: int,
    host: str,
    command_port: int,
    spike_port: int,
    sample_rate_hz: float,
    no_stim: bool = False,
) -> SessionResult:
    """Launch a single session worker subprocess and parse its output."""
    cmd = [
        sys.executable,
        str(_WORKER),
        "--config-path",    config_path,
        "--session-seconds", str(session_seconds),
        "--mode",           mode,
        "--threshold",      str(threshold),
        "--smooth-windows", str(smooth_windows),
        "--host",           host,
        "--command-port",   str(command_port),
        "--spike-port",     str(spike_port),
        "--sample-rate-hz", str(sample_rate_hz),
    ]
    if no_stim:
        cmd.append("--no-stim")

    hard_timeout = session_seconds * 2 + 30  # generous but bounded

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=hard_timeout,
        )
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = f"[experiment_runner] hard timeout after {hard_timeout}s"
    except Exception as exc:
        stdout = ""
        stderr = f"[experiment_runner] subprocess error: {exc}"

    result = parse_log(stdout)

    if result is None:
        # Session crashed or metrics block was not emitted
        if stderr:
            print(f"[experiment_runner] stderr: {stderr.strip()}", file=sys.stderr)
        from autoresearch.metrics import compute_config_hash
        import json
        try:
            config = json.loads(Path(config_path).read_text())
            cfg_hash = compute_config_hash(config)
        except Exception:
            cfg_hash = "unknown"
        from datetime import datetime, timezone
        return SessionResult(
            mean_rally_length=0.0,
            hit_rate=0.0,
            total_rallies=0,
            peak_firing_hz=0.0,
            session_seconds=float(session_seconds),
            config_hash=cfg_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            crashed=True,
        )

    return result
