"""
_training_session_worker.py

Standalone subprocess entry point for a single headless training session.
Called by autoresearch_training.session_runner via subprocess; prints a
standardised metrics block to stdout on clean exit.

Mirrors autoresearch/_session_worker.py but accepts stimulation regime
parameters at runtime (amplitude, duration, n_pulses, isi for reward and
penalty), enabling longitudinal search over the stimulation parameter space.

Usage (invoked by session_runner, not directly by users):
    python -m autoresearch_training._training_session_worker \\
        --config-path /path/to/channel_config.json \\
        --session-seconds 180 \\
        --mode rhx \\
        --regime-id abc1234 \\
        --reward-amplitude-uv 200 \\
        --reward-duration-ms 0.2 \\
        --reward-n-pulses 1 \\
        --reward-isi-ms 10.0 \\
        --penalty-amplitude-uv 300 \\
        --penalty-duration-ms 0.4 \\
        --penalty-n-pulses 1 \\
        --penalty-isi-ms 10.0 \\
        --session-index 0

Stimulator dependency notes
----------------------------
stimulator.py (StimConfig / Stimulator) supports exactly one ManualStimTriggerPulse
per event and configures amplitude/duration at init time only.  Two limitations:

1. Burst stimulation (n_pulses > 1):
   StimConfig has no burst field.  This worker implements burst delivery itself
   by firing ManualStimTriggerPulse n_pulses times with isi_ms sleep between
   pulses using a _BurstStimulator wrapper defined below.

2. Unit conversion (µV → µA):
   StimRegime stores amplitude in µV (search-space label).  stimulator.py
   accepts µA.  This worker passes amplitude_uv directly as amplitude_ua
   (same numeric value, different physical unit).  Physical calibration of the
   µV → µA scale requires electrode impedance measurements; adjust
   _UV_TO_UA_SCALE in this file once impedance is characterised.
   Default _UV_TO_UA_SCALE = 1.0 (1 µV treated as 1 µA numerically).

   Duration conversion: duration_ms × 1000 → duration_us (exact, no calibration needed).
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import socket
import struct
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pong_env import PongEnv
from spike_counter import SpikeCounter
from region_mapper import RegionMapper
from action_decoder import ActionDecoder

# Unit conversion scale factor (see module docstring)
_UV_TO_UA_SCALE: float = 1.0

# RHX spike protocol (matches autoresearch/_session_worker.py)
_SPIKE_MAGIC = 0x3AE2710F
_PACKET_SIZE = 14


# ---------------------------------------------------------------------------
# RHX and mock controllers (identical to autoresearch/_session_worker.py)
# ---------------------------------------------------------------------------

class _RHXController:
    def __init__(
        self,
        host: str,
        command_port: int,
        spike_port: int,
        up_channels: list[int],
        down_channels: list[int],
        threshold: float,
        smooth_windows: int,
        sample_rate_hz: float,
    ) -> None:
        self.host = host
        self.command_port = command_port
        self.spike_port = spike_port
        self.sample_rate_hz = sample_rate_hz

        all_ch = sorted(set(up_channels + down_channels))
        num_channels = max(all_ch) + 1 if all_ch else 1

        self._counter = SpikeCounter(num_channels=num_channels, window_size_s=0.010)
        self._mapper  = RegionMapper(region_channels={"up": up_channels, "down": down_channels})
        self._decoder = ActionDecoder(threshold=threshold, smooth_windows=smooth_windows)

        self.action: str         = "HOLD"
        self.dominance: float    = 0.0
        self.peak_firing_hz: float = 0.0
        self.total_spikes: int   = 0
        self.connected: bool     = False
        self.error: str | None   = None

        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _send_cmd(self, sock: socket.socket, cmd: str) -> str:
        payload = cmd if cmd.endswith(";") else cmd + ";"
        sock.sendall(payload.encode("ascii"))
        time.sleep(0.05)
        try:
            return sock.recv(4096).decode(errors="ignore").strip()
        except socket.timeout:
            return ""

    def _run(self) -> None:
        try:
            self._stream()
        except Exception as exc:
            self.error = str(exc)

    def _stream(self) -> None:
        all_channels = sorted(set(
            self._mapper.region_channels["up"] + self._mapper.region_channels["down"]
        ))

        cmd_sock = socket.create_connection((self.host, self.command_port), timeout=5.0)
        cmd_sock.settimeout(0.15)

        resp = self._send_cmd(cmd_sock, "get RunMode")
        if "RunMode Run" in resp:
            self._send_cmd(cmd_sock, "set RunMode Stop")
            for _ in range(20):
                time.sleep(0.1)
                if "RunMode Stop" in self._send_cmd(cmd_sock, "get RunMode"):
                    break

        self._send_cmd(cmd_sock, f"set TCPSpikeDataOutputHost {self.host}")
        self._send_cmd(cmd_sock, f"set TCPSpikeDataOutputPort {self.spike_port}")
        self._send_cmd(cmd_sock, "execute ClearAllDataOutputs")

        for ch_id in all_channels:
            port_letter = chr(ord("A") + ch_id // 128)
            ch_num = ch_id % 128
            rhx_name = f"{port_letter}-{ch_num:03d}"
            self._send_cmd(cmd_sock, f"set {rhx_name}.TCPDataOutputEnabledSpike true")

        self._send_cmd(cmd_sock, "execute ConnectTCPSpikeDataOutput")
        time.sleep(0.15)
        self._send_cmd(cmd_sock, "set RunMode Run")

        spike_sock = socket.create_connection((self.host, self.spike_port), timeout=5.0)
        spike_sock.settimeout(0.05)
        self.connected = True

        buf = b""
        rhx_time_origin: float | None = None
        wall_offset: float | None     = None

        while not self._stop.is_set():
            try:
                data = spike_sock.recv(8192)
            except socket.timeout:
                data = b""

            t_now = time.time()
            if data:
                buf += data

            while len(buf) >= _PACKET_SIZE:
                chunk, buf = buf[:_PACKET_SIZE], buf[_PACKET_SIZE:]
                magic, raw_name, timestamp, _ = struct.unpack("<I5sIB", chunk)
                if magic != _SPIKE_MAGIC:
                    buf = chunk[1:] + buf
                    break
                name   = raw_name.decode("ascii", errors="ignore").rstrip("\x00")
                ch_int = int(name.split("-")[-1])
                time_s = timestamp / self.sample_rate_hz
                if rhx_time_origin is None:
                    rhx_time_origin = time_s
                    self._counter._next_window_start = time_s
                    wall_offset = time_s - t_now
                self._counter.add_spikes([{"channel": ch_int, "time_s": time_s}])
                self.total_spikes += 1

            if wall_offset is None:
                continue

            current_time_s = t_now + wall_offset
            for window in self._counter.get_window_counts(current_time_s):
                region_window = self._mapper.map(window)
                decision      = self._decoder.decode(region_window)
                self.action    = decision["action"]
                self.dominance = decision["dominance"]
                regions = region_window["regions"]
                hz = max(regions.get("up", 0.0), regions.get("down", 0.0))
                if hz > self.peak_firing_hz:
                    self.peak_firing_hz = hz

        spike_sock.close()
        try:
            self._send_cmd(cmd_sock, "execute DisconnectTCPSpikeDataOutput")
        except Exception:
            pass
        cmd_sock.close()


class _MockController:
    def __init__(self) -> None:
        self.action: str         = "HOLD"
        self.dominance: float    = 0.0
        self.peak_firing_hz: float = 0.0
        self.connected: bool     = True
        self.error: str | None   = None
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            dom = random.gauss(0, 4.0)
            self.dominance = round(dom, 4)
            hz = abs(dom)
            if hz > self.peak_firing_hz:
                self.peak_firing_hz = hz
            if dom > 3.0:
                self.action = "UP"
            elif dom < -3.0:
                self.action = "DOWN"
            else:
                self.action = "HOLD"
            time.sleep(0.010)


# ---------------------------------------------------------------------------
# Burst stimulator
# ---------------------------------------------------------------------------

class _BurstStimulator:
    """
    Stimulator that supports burst delivery (n_pulses > 1 per event).

    Wraps the RHX command socket directly.  For n_pulses == 1, behaviour is
    identical to the standard Stimulator.  For n_pulses > 1, fires
    ManualStimTriggerPulse n_pulses times with isi_ms sleep between each.

    Limitation: stimulator.py's Stimulator class configures amplitude/duration
    at init time and fires a single pulse per event.  It does not natively
    support burst delivery.  _BurstStimulator reimplements the trigger loop
    to add burst support while reusing StimConfig for channel configuration.

    Required minimal change to stimulator.py (not made here — flagged for human):
        Add optional burst_n_pulses: int = 1 and burst_isi_ms: float = 10.0
        fields to StimConfig, then loop ManualStimTriggerPulse n times in
        Stimulator._trigger().  This worker provides a standalone implementation
        in the meantime.
    """

    def __init__(
        self,
        host: str,
        command_port: int,
        reward_channels: list[int],
        penalty_channels: list[int],
        reward_amplitude_uv: float,
        reward_duration_ms: float,
        reward_n_pulses: int,
        reward_isi_ms: float,
        penalty_amplitude_uv: float,
        penalty_duration_ms: float,
        penalty_n_pulses: int,
        penalty_isi_ms: float,
        refractory_ms: float = 200.0,
    ) -> None:
        self._host           = host
        self._command_port   = command_port
        self._reward_ch      = reward_channels
        self._penalty_ch     = penalty_channels

        # Convert µV → µA (same numeric value, see module-level _UV_TO_UA_SCALE)
        self._reward_amp_ua  = int(reward_amplitude_uv  * _UV_TO_UA_SCALE)
        self._reward_dur_us  = int(reward_duration_ms   * 1000)
        self._reward_n       = reward_n_pulses
        self._reward_isi_s   = reward_isi_ms / 1000.0

        self._penalty_amp_ua = int(penalty_amplitude_uv * _UV_TO_UA_SCALE)
        self._penalty_dur_us = int(penalty_duration_ms  * 1000)
        self._penalty_n      = penalty_n_pulses
        self._penalty_isi_s  = penalty_isi_ms / 1000.0

        self._refractory_s   = refractory_ms / 1000.0
        self._last_stim: float = 0.0

        self._queue: queue.Queue[str] = queue.Queue()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._sock = socket.create_connection((self._host, self._command_port), timeout=5.0)
        self._sock.settimeout(0.15)
        self._configure()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def on_event(self, event: str) -> None:
        if event in ("hit", "miss", "score"):
            self._queue.put(event)

    def _send(self, cmd: str) -> None:
        assert self._sock is not None
        payload = cmd if cmd.endswith(";") else cmd + ";"
        with self._lock:
            self._sock.sendall(payload.encode("ascii"))
            time.sleep(0.04)
            try:
                self._sock.recv(4096)
            except socket.timeout:
                pass

    def _ch_name(self, ch: int) -> str:
        return f"{chr(ord('A') + ch // 128)}-{ch % 128:03d}"

    def _configure(self) -> None:
        all_ch = sorted(set(self._reward_ch + self._penalty_ch))
        for ch in all_ch:
            name = self._ch_name(ch)
            self._send(f"set {name}.StimEnabled true")
            self._send(f"set {name}.StimShape Biphasic")
            self._send(f"set {name}.StimPolarity NegativeFirst")

        for ch in self._reward_ch:
            name = self._ch_name(ch)
            self._send(f"set {name}.FirstPhaseDurationMicroseconds {self._reward_dur_us}")
            self._send(f"set {name}.FirstPhaseAmplitudeMicroamps {self._reward_amp_ua}")
            self._send(f"set {name}.SecondPhaseDurationMicroseconds {self._reward_dur_us}")
            self._send(f"set {name}.SecondPhaseAmplitudeMicroamps {self._reward_amp_ua}")

        for ch in self._penalty_ch:
            name = self._ch_name(ch)
            self._send(f"set {name}.FirstPhaseDurationMicroseconds {self._penalty_dur_us}")
            self._send(f"set {name}.FirstPhaseAmplitudeMicroamps {self._penalty_amp_ua}")
            self._send(f"set {name}.SecondPhaseDurationMicroseconds {self._penalty_dur_us}")
            self._send(f"set {name}.SecondPhaseAmplitudeMicroamps {self._penalty_amp_ua}")

    def _burst(self, channels: list[int], n_pulses: int, isi_s: float) -> None:
        for pulse_i in range(n_pulses):
            for ch in channels:
                self._send(f"execute ManualStimTriggerPulse {self._ch_name(ch)}")
            if pulse_i < n_pulses - 1:
                time.sleep(isi_s)

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            now = time.monotonic()
            if (now - self._last_stim) < self._refractory_s:
                continue

            self._last_stim = now

            if event in ("hit", "score") and self._reward_ch:
                self._burst(self._reward_ch, self._reward_n, self._reward_isi_s)
            elif event == "miss" and self._penalty_ch:
                self._burst(self._penalty_ch, self._penalty_n, self._penalty_isi_s)


# ---------------------------------------------------------------------------
# Main session runner
# ---------------------------------------------------------------------------

def _run_session(args: argparse.Namespace) -> None:
    config   = json.loads(Path(args.config_path).read_text())
    up_ch:   list[int] = config.get("up_channels", [])
    down_ch: list[int] = config.get("down_channels", [])

    if args.mode == "mock":
        ctrl: _MockController | _RHXController = _MockController()
    else:
        ctrl = _RHXController(
            host=args.host,
            command_port=args.command_port,
            spike_port=args.spike_port,
            up_channels=up_ch,
            down_channels=down_ch,
            threshold=args.threshold,
            smooth_windows=args.smooth_windows,
            sample_rate_hz=args.sample_rate_hz,
        )

    ctrl.start()

    if args.mode == "rhx":
        deadline = time.time() + 10.0
        while not ctrl.connected and ctrl.error is None and time.time() < deadline:
            time.sleep(0.1)
        if ctrl.error:
            print(f"[training_worker] RHX connection error: {ctrl.error}", file=sys.stderr)
            ctrl.stop()
            sys.exit(1)
        if not ctrl.connected:
            print("[training_worker] Timed out waiting for RHX connection.", file=sys.stderr)
            ctrl.stop()
            sys.exit(1)

    env = PongEnv(paddle_speed=6.0, ai_speed=4.5, ball_speed_init=5.0)
    env.reset()

    # Wire up stimulation if enabled and in RHX mode
    stim: _BurstStimulator | None = None
    if args.mode == "rhx" and args.stim_enabled:
        stim_ch: list[int] = config.get("stim_channels", [])
        if stim_ch:
            mid = len(stim_ch) // 2
            reward_ch  = stim_ch[:mid] or stim_ch
            penalty_ch = stim_ch[mid:] or stim_ch
            try:
                stim = _BurstStimulator(
                    host=args.host,
                    command_port=args.command_port,
                    reward_channels=reward_ch,
                    penalty_channels=penalty_ch,
                    reward_amplitude_uv=args.reward_amplitude_uv,
                    reward_duration_ms=args.reward_duration_ms,
                    reward_n_pulses=args.reward_n_pulses,
                    reward_isi_ms=args.reward_isi_ms,
                    penalty_amplitude_uv=args.penalty_amplitude_uv,
                    penalty_duration_ms=args.penalty_duration_ms,
                    penalty_n_pulses=args.penalty_n_pulses,
                    penalty_isi_ms=args.penalty_isi_ms,
                )
                stim.start()
            except Exception as exc:
                print(
                    f"[training_worker] stim setup failed (continuing without): {exc}",
                    file=sys.stderr,
                )
                stim = None

    # Session loop
    total_hit_events: int    = 0
    total_miss_events: int   = 0
    current_rally_hits: int  = 0
    rally_lengths: list[int] = []

    t_start = time.time()
    t_end   = t_start + args.session_seconds
    frame_s = 1.0 / 60.0

    while time.time() < t_end:
        frame_start = time.time()
        state = env.step(ctrl.action)

        if state.event == "hit":
            total_hit_events   += 1
            current_rally_hits += 1
        elif state.event in ("miss", "score"):
            rally_lengths.append(current_rally_hits)
            current_rally_hits = 0
            if state.event == "miss":
                total_miss_events += 1

        if stim and state.event:
            stim.on_event(state.event)

        elapsed = time.time() - frame_start
        if elapsed < frame_s:
            time.sleep(frame_s - elapsed)

    if stim:
        stim.stop()
    ctrl.stop()

    if current_rally_hits > 0:
        rally_lengths.append(current_rally_hits)

    session_seconds = time.time() - t_start
    total_rallies   = len(rally_lengths)
    mean_rally      = sum(rally_lengths) / max(total_rallies, 1)
    denom           = total_hit_events + total_miss_events
    hit_rate        = total_hit_events / denom if denom > 0 else 0.0
    peak_hz         = ctrl.peak_firing_hz

    # Emit standardised metrics block.
    # config_hash is set to regime_id for compatibility with autoresearch.metrics.parse_log().
    print("---")
    print(f"mean_rally_length: {mean_rally:.4f}")
    print(f"hit_rate:          {hit_rate:.4f}")
    print(f"total_rallies:     {total_rallies}")
    print(f"peak_firing_hz:    {peak_hz:.4f}")
    print(f"session_seconds:   {session_seconds:.2f}")
    print(f"config_hash:       {args.regime_id}")   # alias for parse_log compat
    print(f"regime_id:         {args.regime_id}")
    print(f"session_index:     {args.session_index}")
    print(f"stim_enabled:      {str(args.stim_enabled).lower()}")
    sys.stdout.flush()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless training Pong session worker")
    p.add_argument("--config-path",          required=True)
    p.add_argument("--session-seconds",      type=float, default=180.0)
    p.add_argument("--mode",                 choices=["rhx", "mock"], default="rhx")
    p.add_argument("--threshold",            type=float, default=3.0)
    p.add_argument("--smooth-windows",       type=int,   default=1)
    p.add_argument("--host",                 default="127.0.0.1")
    p.add_argument("--command-port",         type=int,   default=5000)
    p.add_argument("--spike-port",           type=int,   default=5002)
    p.add_argument("--sample-rate-hz",       type=float, default=30000.0)
    p.add_argument("--regime-id",            default="unknown")
    p.add_argument("--session-index",        type=int,   default=0)
    p.add_argument("--stim-enabled",         action="store_true", default=True)
    p.add_argument("--no-stim",              dest="stim_enabled", action="store_false")
    # Reward pulse parameters
    p.add_argument("--reward-amplitude-uv",  type=float, default=200.0)
    p.add_argument("--reward-duration-ms",   type=float, default=0.2)
    p.add_argument("--reward-n-pulses",      type=int,   default=1)
    p.add_argument("--reward-isi-ms",        type=float, default=10.0)
    # Penalty pulse parameters
    p.add_argument("--penalty-amplitude-uv", type=float, default=400.0)
    p.add_argument("--penalty-duration-ms",  type=float, default=0.4)
    p.add_argument("--penalty-n-pulses",     type=int,   default=1)
    p.add_argument("--penalty-isi-ms",       type=float, default=10.0)
    return p


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    _run_session(_args)
