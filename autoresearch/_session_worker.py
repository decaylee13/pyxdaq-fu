"""
_session_worker.py

Standalone subprocess entry point for a single headless neural Pong session.
Called by experiment_runner.py via subprocess; prints a standardised metrics
block to stdout on clean exit so the parent process can parse results.

This module intentionally does NOT import pygame.  It drives the game
physics engine (PongEnv) directly and connects to RHX hardware using the
same TCP protocol that pong_game.py's NeuralController uses.

Usage (invoked by experiment_runner, not directly by users):
    python -m autoresearch._session_worker \\
        --config-path /path/to/channel_config.json \\
        --session-seconds 90 \\
        --mode rhx \\
        --threshold 3.0 \\
        --smooth-windows 1 \\
        --host 127.0.0.1 \\
        --command-port 5000 \\
        --spike-port 5002
"""

from __future__ import annotations

import argparse
import json
import math
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

# RHX spike protocol constants (matches pong_game.py)
_SPIKE_MAGIC = 0x3AE2710F
_PACKET_SIZE = 14


# ---------------------------------------------------------------------------
# Minimal RHX neural controller (no pygame, no display)
# ---------------------------------------------------------------------------

class _RHXController:
    """
    Background-thread RHX spike reader, identical in logic to NeuralController
    in pong_game.py but without any pygame dependency.
    Exposes: .action, .dominance, .peak_firing_hz, .connected, .error
    """

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
        self._mapper = RegionMapper(region_channels={"up": up_channels, "down": down_channels})
        self._decoder = ActionDecoder(threshold=threshold, smooth_windows=smooth_windows)

        self.action: str = "HOLD"
        self.dominance: float = 0.0
        self.peak_firing_hz: float = 0.0
        self.total_spikes: int = 0
        self.connected: bool = False
        self.error: str | None = None

        self._stop = threading.Event()
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
        wall_offset: float | None = None

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
                name = raw_name.decode("ascii", errors="ignore").rstrip("\x00")
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
                decision = self._decoder.decode(region_window)
                self.action = decision["action"]
                self.dominance = decision["dominance"]
                # track peak region activity for culture health metric
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


# ---------------------------------------------------------------------------
# Mock controller (for --mode mock / unit testing)
# ---------------------------------------------------------------------------

class _MockController:
    """
    Simulates neural activity with random UP/DOWN decisions.
    Used when --mode mock is passed (no hardware required).
    """

    def __init__(self) -> None:
        self.action: str = "HOLD"
        self.dominance: float = 0.0
        self.peak_firing_hz: float = 0.0
        self.connected: bool = True
        self.error: str | None = None
        self._stop = threading.Event()
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
# Main session runner
# ---------------------------------------------------------------------------

def _run_session(args: argparse.Namespace) -> None:
    # Load config for metadata (hash only — don't validate here)
    config = json.loads(Path(args.config_path).read_text())
    from autoresearch.metrics import compute_config_hash
    cfg_hash = compute_config_hash(config)

    up_channels: list[int] = config.get("up_channels", [])
    down_channels: list[int] = config.get("down_channels", [])

    # Build controller
    if args.mode == "mock":
        ctrl: _MockController | _RHXController = _MockController()
    else:
        ctrl = _RHXController(
            host=args.host,
            command_port=args.command_port,
            spike_port=args.spike_port,
            up_channels=up_channels,
            down_channels=down_channels,
            threshold=args.threshold,
            smooth_windows=args.smooth_windows,
            sample_rate_hz=args.sample_rate_hz,
        )

    ctrl.start()

    # Wait for RHX connection (up to 10 s)
    if args.mode == "rhx":
        deadline = time.time() + 10.0
        while not ctrl.connected and ctrl.error is None and time.time() < deadline:
            time.sleep(0.1)
        if ctrl.error:
            print(f"[session_worker] RHX connection error: {ctrl.error}", file=sys.stderr)
            ctrl.stop()
            sys.exit(1)
        if not ctrl.connected:
            print("[session_worker] Timed out waiting for RHX connection.", file=sys.stderr)
            ctrl.stop()
            sys.exit(1)

    env = PongEnv(paddle_speed=6.0, ai_speed=4.5, ball_speed_init=5.0)
    env.reset()

    # Optionally wire up stimulation (RHX mode only, not suppressed by --no-stim)
    stim = None
    if args.mode == "rhx" and not getattr(args, "no_stim", False):
        stim_channels: list[int] = config.get("stim_channels", [])
        if stim_channels:
            try:
                import threading
                from stimulator import Stimulator, StimConfig
                mid = len(stim_channels) // 2
                stim_cfg = StimConfig(
                    reward_channels=stim_channels[:mid] or stim_channels,
                    penalty_channels=stim_channels[mid:] or stim_channels,
                    reward_duration_us=200,
                    reward_amplitude_ua=10,
                    penalty_duration_us=400,
                    penalty_amplitude_ua=20,
                    refractory_ms=200.0,
                )
                stim_sock = socket.create_connection(
                    (args.host, args.command_port), timeout=5.0
                )
                stim_sock.settimeout(0.15)
                stim = Stimulator(
                    stim_sock, stim_cfg, threading.Lock(), verbose=False
                )
                stim._sock = stim_sock
                stim.start()
            except Exception as exc:
                print(f"[session_worker] stim setup failed (continuing without): {exc}",
                      file=sys.stderr)
                stim = None

    # Metric accumulators
    total_hit_events: int = 0
    total_miss_events: int = 0
    current_rally_hits: int = 0
    rally_lengths: list[int] = []

    t_start = time.time()
    t_end = t_start + args.session_seconds
    frame_s = 1.0 / 60.0

    while time.time() < t_end:
        frame_start = time.time()

        state = env.step(ctrl.action)

        if state.event == "hit":
            total_hit_events += 1
            current_rally_hits += 1
        elif state.event in ("miss", "score"):
            rally_lengths.append(current_rally_hits)
            current_rally_hits = 0
            if state.event == "miss":
                total_miss_events += 1

        if stim and state.event:
            stim.on_event(state.event)

        # Pace to ~60 FPS
        elapsed = time.time() - frame_start
        if elapsed < frame_s:
            time.sleep(frame_s - elapsed)

    if stim:
        stim.stop()
    ctrl.stop()

    # Handle open rally at session end
    if current_rally_hits > 0:
        rally_lengths.append(current_rally_hits)

    session_seconds = time.time() - t_start
    total_rallies = len(rally_lengths)
    mean_rally = sum(rally_lengths) / max(total_rallies, 1)
    denom = total_hit_events + total_miss_events
    hit_rate = total_hit_events / denom if denom > 0 else 0.0
    peak_hz = ctrl.peak_firing_hz

    # Emit standardised metrics block — parsed by metrics.parse_log()
    print("---")
    print(f"mean_rally_length: {mean_rally:.4f}")
    print(f"hit_rate:          {hit_rate:.4f}")
    print(f"total_rallies:     {total_rallies}")
    print(f"peak_firing_hz:    {peak_hz:.4f}")
    print(f"session_seconds:   {session_seconds:.2f}")
    print(f"config_hash:       {cfg_hash}")
    sys.stdout.flush()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless neural Pong session worker")
    p.add_argument("--config-path",    required=True)
    p.add_argument("--session-seconds", type=float, default=90.0)
    p.add_argument("--mode",           choices=["rhx", "mock"], default="rhx")
    p.add_argument("--threshold",      type=float, default=3.0)
    p.add_argument("--smooth-windows", type=int,   default=1)
    p.add_argument("--host",           default="127.0.0.1")
    p.add_argument("--command-port",   type=int,   default=5000)
    p.add_argument("--spike-port",     type=int,   default=5002)
    p.add_argument("--sample-rate-hz", type=float, default=30000.0)
    p.add_argument("--no-stim",        action="store_true",
                   help="Disable stimulation feedback (mirrors pong_game.py --no-stim)")
    return p


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    _run_session(_args)
