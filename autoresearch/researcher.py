"""
researcher.py

The AutoResearcher class — the top-level orchestrator for the autonomous
experiment loop.  It reads program.md for research context, runs the
perturbation-evaluate-keep/revert cycle, manages git checkpoints, detects
culture drift, and surfaces a synthesis of all results on demand.
"""

from __future__ import annotations

import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

from autoresearch.config_manager import ConfigManager
from autoresearch.constraints import (
    MIN_DOWN_CHANNELS,
    MIN_UP_CHANNELS,
    MIN_STIM_REC_SEPARATION,
    ConstraintViolationError,
    _chebyshev,
    _id_to_rc,
)
from autoresearch.experiment_runner import run_session
from autoresearch.metrics import SessionResult, compute_config_hash
from autoresearch.perturbations import (
    activate_channel,
    add_stim_channel,
    adjust_dominance_threshold,
    adjust_hold_deadband,
    describe_perturbation,
    silence_channel,
    swap_region,
    swap_stim_rec,
)
from autoresearch.results_logger import ResultsLogger


class AutoResearcher:
    """
    Autonomous experiment loop for optimising electrode configurations.

    Reads program.md for human-supplied research context, runs the
    perturbation → session → keep/revert loop, and manages git state
    so every accepted config is a recoverable checkpoint.
    """

    def __init__(
        self,
        project_root: str,
        config_path: Optional[str] = None,
        results_path: Optional[str] = None,
        session_seconds: int = 90,
        n_repeats: int = 2,
        drift_check_interval: int = 10,
        min_peak_firing_hz: float = 5.0,
        mode: str = "rhx",
        threshold: float = 3.0,
        smooth_windows: int = 1,
        host: str = "127.0.0.1",
        command_port: int = 5000,
        spike_port: int = 5002,
        excluded_channels: Optional[list[tuple[int, int]]] = None,
        no_stim: bool = False,
    ) -> None:
        self._root = Path(project_root).resolve()
        self._config_path = str(
            Path(config_path) if config_path else self._root / "channel_config.json"
        )
        results_tsv = (
            Path(results_path)
            if results_path
            else Path(__file__).parent / "results.tsv"
        )
        self._cfg = ConfigManager(self._config_path)
        self._logger = ResultsLogger(str(results_tsv))

        self._session_seconds = session_seconds
        self._n_repeats = n_repeats
        self._drift_check_interval = drift_check_interval
        self._min_peak_firing_hz = min_peak_firing_hz
        self._mode = mode
        self._threshold = threshold
        self._smooth_windows = smooth_windows
        self._host = host
        self._command_port = command_port
        self._spike_port = spike_port
        self._no_stim = no_stim
        # Set of (row, col) channel coordinates that autoresearch will never
        # promote or swap — corresponds to dead/unreliable channels documented
        # in program.md.
        self._excluded: set[tuple[int, int]] = set(excluded_channels or [])

        self._experiment_count: int = 0
        self._consecutive_tier2: int = 0
        self._baseline_score: float = 0.0
        self._decoder_params: dict = {
            "threshold": threshold,
            "smooth_windows": smooth_windows,
        }
        self._program_md = self._load_program_md()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Run once before the main loop:
        - Validate the current config (with guidance on failure)
        - Check RHX hardware connection
        - Run a baseline experiment and log it
        - Print a summary of the current electrode layout
        """
        print("\n=== AutoResearcher setup ===")
        print(f"Project root : {self._root}")
        print(f"Config       : {self._config_path}")
        print(f"Mode         : {self._mode}")

        # Print the research objective from program.md (first non-empty lines)
        if self._program_md:
            obj_lines: list[str] = []
            in_obj = False
            for line in self._program_md.splitlines():
                if line.startswith("## Objective"):
                    in_obj = True
                    continue
                if in_obj:
                    if line.startswith("##"):
                        break
                    if line.strip():
                        obj_lines.append(f"  {line.strip()}")
            if obj_lines:
                print("Objective    :", obj_lines[0].strip())
                for l in obj_lines[1:3]:
                    print("              ", l.strip())

        # Validate config
        try:
            config = self._cfg.load()
        except ConstraintViolationError as exc:
            print(
                f"\n[setup] Config validation failed:\n  {exc}\n"
                "  Run channel_selector.py to assign UP/DOWN regions before "
                "starting autoresearch."
            )
            raise
        except FileNotFoundError:
            print(f"\n[setup] Config file not found: {self._config_path}")
            raise

        # Print layout summary
        self._print_layout(config)

        # Hardware check
        if self._mode == "rhx":
            if not self._check_hardware():
                raise RuntimeError(
                    f"Cannot reach RHX at {self._host}:{self._command_port}. "
                    "Ensure RHX software is running before calling setup()."
                )
            print("[setup] RHX hardware connection OK.")

        # Backup the starting config
        bak = self._cfg.backup()
        print(f"[setup] Config backed up to {bak}")

        # Baseline experiment
        print(f"[setup] Running baseline ({self._n_repeats}x {self._session_seconds}s)…")
        baseline = self._run_session_now()
        self._baseline_score = baseline.mean_rally_length
        commit = self._git_commit("autoresearch: baseline")
        self._logger.log(commit, baseline, "keep", "baseline")

        print(
            f"[setup] Baseline mean_rally_length = {baseline.mean_rally_length:.3f}  "
            f"hit_rate = {baseline.hit_rate:.3f}  "
            f"peak_hz = {baseline.peak_firing_hz:.1f}"
        )
        if baseline.peak_firing_hz < self._min_peak_firing_hz:
            print(
                f"[setup] WARNING: peak firing rate {baseline.peak_firing_hz:.1f} Hz < "
                f"floor {self._min_peak_firing_hz} Hz — culture may be unhealthy."
            )
        print("=== setup complete ===\n")

    def run(self, max_experiments: Optional[int] = None) -> None:
        """
        Main experiment loop.  Runs until max_experiments is reached or
        KeyboardInterrupt.  Automatically triggers drift re-checks every
        drift_check_interval accepted experiments.
        """
        print(f"[run] Starting loop (max_experiments={max_experiments or '∞'})")
        try:
            while True:
                if max_experiments is not None and self._experiment_count >= max_experiments:
                    print(f"[run] Reached max_experiments={max_experiments}. Stopping.")
                    break

                perturbation_fn, kwargs = self._select_perturbation()
                kept = self._run_one_experiment(perturbation_fn, **kwargs)

                if kept:
                    kept_count = sum(
                        1 for r in self._logger.get_recent(200)
                        if r["status"] == "keep"
                    )
                    if kept_count > 0 and kept_count % self._drift_check_interval == 0:
                        self._run_drift_check()

        except KeyboardInterrupt:
            print("\n[run] Interrupted by user.")

        print(f"[run] Done — {self._experiment_count} experiments total.")

    def _run_one_experiment(self, perturbation_fn, **kwargs) -> bool:
        """
        Apply one perturbation, run a session, log result, keep or revert.
        Returns True if the result was accepted (kept).
        """
        self._experiment_count += 1

        # Load current config
        try:
            config_before = self._cfg.load()
        except ConstraintViolationError as exc:
            print(f"[exp {self._experiment_count}] Invalid current config: {exc}")
            return False

        # Apply perturbation
        try:
            if perturbation_fn in (adjust_dominance_threshold, adjust_hold_deadband):
                # Decoder-only perturbation
                params_after = perturbation_fn(self._decoder_params, **kwargs)
                config_after = config_before
                before_thresh = self._decoder_params.get("threshold", "?")
                after_thresh = params_after.get("threshold", "?")
                description = (
                    f"decoder threshold {before_thresh}→{after_thresh} Hz "
                    f"via {perturbation_fn.__name__}"
                )
            else:
                config_after = perturbation_fn(config_before, **kwargs)
                params_after = self._decoder_params
                description = describe_perturbation(config_before, config_after)
        except ConstraintViolationError as exc:
            print(f"[exp {self._experiment_count}] Perturbation rejected: {exc}")
            return False

        # Write new config
        try:
            self._cfg.save(config_after)
        except ConstraintViolationError as exc:
            print(f"[exp {self._experiment_count}] Save rejected: {exc}")
            return False

        # Run session
        result = self._run_session_now(
            threshold=params_after.get("threshold", self._threshold),
            smooth_windows=params_after.get("smooth_windows", self._smooth_windows),
        )

        if result.crashed:
            print(f"[exp {self._experiment_count}] CRASH — reverting.")
            self._git_revert()
            commit = self._git_commit("autoresearch: revert after crash")
            self._logger.log(commit, result, "crash", description)
            return False

        # Culture health floor
        if result.peak_firing_hz < self._min_peak_firing_hz:
            print(
                f"[exp {self._experiment_count}] peak_hz {result.peak_firing_hz:.1f} < "
                f"floor {self._min_peak_firing_hz} — culture may be degrading."
            )

        # Keep or discard
        best = self._logger.get_best()
        best_score = best[1].mean_rally_length if best else self._baseline_score
        new_score = result.mean_rally_length

        if new_score >= best_score:
            status = "keep"
            self._decoder_params = params_after
            commit = self._git_commit(
                f"autoresearch: keep exp#{self._experiment_count} "
                f"rally={new_score:.3f} ({description[:60]})"
            )
            print(
                f"[exp {self._experiment_count}] KEEP  rally={new_score:.3f} "
                f"(was {best_score:.3f})  {description}"
            )
        else:
            status = "discard"
            self._git_revert()
            # Restore previous decoder params
            commit = self._git_commit(
                f"autoresearch: discard exp#{self._experiment_count}"
            )
            print(
                f"[exp {self._experiment_count}] DISCARD rally={new_score:.3f} "
                f"(best {best_score:.3f})  {description}"
            )

        self._logger.log(commit, result, status, description)

        # Monotonic decline detection
        if self._logger.detect_monotonic_decline(window=5):
            print(
                "[run] WARNING: 5 consecutive declined experiments — "
                "consider checking culture health or adjusting strategy."
            )

        if self._experiment_count % 20 == 0:
            self._logger.write_summary(self.summarize())

        return status == "keep"

    def _check_hardware(self) -> bool:
        """Try to open a TCP connection to the RHX command port."""
        try:
            with socket.create_connection(
                (self._host, self._command_port), timeout=3.0
            ):
                return True
        except OSError:
            return False

    def _run_drift_check(self) -> None:
        """
        Restore the best-known config, run a session, warn if the score has
        degraded by >15% vs the original baseline, then restore the working config.
        """
        best = self._logger.get_best()
        if best is None:
            return

        best_commit, best_result = best
        working_config = self._cfg.load_raw()  # save current state

        print(f"[drift] Running drift check against best commit {best_commit}…")

        # Restore best config via git checkout
        subprocess.run(
            ["git", "checkout", best_commit, "--", Path(self._config_path).name],
            cwd=self._root,
            capture_output=True,
        )

        drift_result = self._run_session_now()
        ratio = (
            drift_result.mean_rally_length / best_result.mean_rally_length
            if best_result.mean_rally_length > 0
            else 1.0
        )

        if ratio < 0.85:
            print(
                f"[drift] WARNING: performance degraded to {ratio:.1%} of best "
                f"({drift_result.mean_rally_length:.3f} vs {best_result.mean_rally_length:.3f}). "
                "Culture may be drifting."
            )
        else:
            print(f"[drift] OK — drift check score {drift_result.mean_rally_length:.3f} ({ratio:.1%})")

        # Restore working config
        self._cfg.save(working_config)

    def _select_perturbation(self) -> tuple[callable, dict]:
        """
        Stochastically choose the next perturbation and valid arguments.

        Tier 1 (channel swaps):   70% of the time
        Tier 2 (stim):            20%
        Tier 3 (decoder params):  10%
        Tier 2 is capped at 3 consecutive uses.
        """
        config = self._cfg.load_raw()
        cols = config.get("grid_cols", 4)

        r = random.random()

        # Force out of Tier 2 if used 3 times in a row
        if self._consecutive_tier2 >= 3:
            tier = 1 if r < 0.875 else 3
        elif r < 0.70:
            tier = 1
        elif r < 0.90:
            tier = 2
        else:
            tier = 3

        if tier == 1:
            self._consecutive_tier2 = 0
            return self._pick_tier1(config, cols)
        elif tier == 2:
            self._consecutive_tier2 += 1
            result = self._pick_tier2(config, cols)
            if result is not None:
                return result
            # Fall back to tier 1 if no valid tier-2 move exists
            self._consecutive_tier2 = 0
            return self._pick_tier1(config, cols)
        else:
            self._consecutive_tier2 = 0
            return self._pick_tier3()

    def _pick_tier1(self, config: dict, cols: int) -> tuple[callable, dict]:
        """Randomly pick a valid Tier 1 (channel role swap) perturbation."""
        up_ids: list[int] = config.get("up_channels", [])
        down_ids: list[int] = config.get("down_channels", [])
        off_ids: list[int] = config.get("disabled_channels", [])

        rec_ids = up_ids + down_ids

        choices: list[tuple[callable, dict]] = []

        # swap_region: any UP or DOWN channel that is not excluded
        for ch_id in rec_ids:
            rc = _id_to_rc(ch_id, cols)
            if rc in self._excluded:
                continue
            choices.append((swap_region, {"channel": rc}))

        # silence_channel: only if region minimums would still be met after
        for ch_id in rec_ids:
            rc = _id_to_rc(ch_id, cols)
            if rc in self._excluded:
                continue
            region = "UP" if ch_id in up_ids else "DOWN"
            region_size = len(up_ids) if region == "UP" else len(down_ids)
            min_size = MIN_UP_CHANNELS if region == "UP" else MIN_DOWN_CHANNELS
            if region_size > min_size:
                choices.append((silence_channel, {"channel": rc}))

        # activate_channel: any OFF channel that is not excluded
        for ch_id in off_ids:
            rc = _id_to_rc(ch_id, cols)
            if rc in self._excluded:
                continue
            choices.append((activate_channel, {"channel": rc}))

        if not choices:
            # Absolute fallback: swap a random non-excluded rec channel
            candidates = [
                ch for ch in rec_ids if _id_to_rc(ch, cols) not in self._excluded
            ]
            if candidates:
                ch_id = random.choice(candidates)
                return (swap_region, {"channel": _id_to_rc(ch_id, cols)})
            # Nothing to do — fall through to a decoder param adjustment
            return self._pick_tier3()

        fn, kwargs = random.choice(choices)
        return (fn, kwargs)

    def _pick_tier2(
        self, config: dict, cols: int
    ) -> Optional[tuple[callable, dict]]:
        """Randomly pick a valid Tier 2 (stim) perturbation, or None."""
        stim_ids: list[int] = config.get("stim_channels", [])
        up_ids: list[int] = config.get("up_channels", [])
        down_ids: list[int] = config.get("down_channels", [])
        rec_ids = up_ids + down_ids

        choices: list[tuple[callable, dict]] = []

        # swap_stim_rec: pick a (stim, rec) pair — skip excluded channels
        if stim_ids and rec_ids:
            for s in stim_ids:
                s_rc = _id_to_rc(s, cols)
                if s_rc in self._excluded:
                    continue
                for r in rec_ids:
                    r_rc = _id_to_rc(r, cols)
                    if r_rc in self._excluded:
                        continue
                    if _chebyshev(s_rc, r_rc) <= MIN_STIM_REC_SEPARATION + 1:
                        choices.append(
                            (swap_stim_rec, {"stim_ch": s_rc, "rec_ch": r_rc})
                        )

        if choices:
            fn, kwargs = random.choice(choices)
            return (fn, kwargs)
        return None

    def _pick_tier3(self) -> tuple[callable, dict]:
        """Pick a Tier 3 (decoder param) perturbation."""
        if random.random() < 0.5:
            delta = random.choice([-0.15, -0.10, +0.10, +0.15])
            return (adjust_dominance_threshold, {"delta_pct": delta})
        else:
            delta = random.choice([-1.0, -0.5, +0.5, +1.0])
            return (adjust_hold_deadband, {"delta_hz": delta})

    def _git_commit(self, message: str) -> str:
        """Stage channel_config.json and commit. Returns short hash."""
        subprocess.run(
            ["git", "add", Path(self._config_path).name],
            cwd=self._root,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", message],
            cwd=self._root,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=self._root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _git_revert(self) -> None:
        """Hard-reset channel_config.json to HEAD (discard uncommitted changes)."""
        subprocess.run(
            ["git", "checkout", "--", Path(self._config_path).name],
            cwd=self._root,
            capture_output=True,
        )

    def _run_session_now(
        self,
        threshold: Optional[float] = None,
        smooth_windows: Optional[int] = None,
    ) -> SessionResult:
        """Run a session with current settings."""
        return run_session(
            config_path=self._config_path,
            session_seconds=self._session_seconds,
            n_repeats=self._n_repeats,
            mode=self._mode,
            threshold=threshold if threshold is not None else self._threshold,
            smooth_windows=smooth_windows if smooth_windows is not None else self._smooth_windows,
            host=self._host,
            command_port=self._command_port,
            spike_port=self._spike_port,
            no_stim=self._no_stim,
        )

    def summarize(self) -> str:
        """
        Return a synthesis of all experiments so far including the best
        config, patterns, and drift events.
        """
        best = self._logger.get_best()
        recent = self._logger.get_recent(20)
        kept = [r for r in recent if r["status"] == "keep"]
        discarded = [r for r in recent if r["status"] == "discard"]
        crashes = [r for r in recent if r["status"] == "crash"]

        lines = [
            f"AutoResearcher Summary — {self._experiment_count} experiments",
            "=" * 60,
        ]

        if best:
            lines.append(
                f"Best config  : commit {best[0]}  "
                f"mean_rally_length={best[1].mean_rally_length:.3f}"
            )
        else:
            lines.append("Best config  : none yet")

        lines.append(
            f"Recent (last 20): {len(kept)} kept, "
            f"{len(discarded)} discarded, {len(crashes)} crashes"
        )

        if self._logger.detect_monotonic_decline(window=5):
            lines.append(
                "DRIFT WARNING: 5 consecutive declines detected — "
                "culture may be degrading."
            )

        lines.append(f"Baseline score : {self._baseline_score:.3f}")
        lines.append(f"Decoder params : {self._decoder_params}")

        lines.append("")
        lines.append("Recent kept experiments:")
        for r in kept[-5:]:
            lines.append(
                f"  commit={r['commit']}  "
                f"rally={float(r['mean_rally_length']):.3f}  "
                f"{r['description']}"
            )

        summary = "\n".join(lines)
        print(summary)
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_program_md(self) -> str:
        md_path = Path(__file__).parent / "program.md"
        if md_path.exists():
            return md_path.read_text()
        return ""

    def _print_layout(self, config: dict) -> None:
        """Print a visual ASCII grid of channel roles."""
        rows = config.get("grid_rows", 8)
        cols = config.get("grid_cols", 4)
        up = set(config.get("up_channels", []))
        down = set(config.get("down_channels", []))
        stim = set(config.get("stim_channels", []))
        off = set(config.get("disabled_channels", []))

        print(
            f"\nElectrode layout ({rows}×{cols}):  "
            "U=UP  D=DOWN  S=STIM  .=OFF  X=excluded  ?=unassigned"
        )
        if self._excluded:
            print(f"  Excluded from search: {sorted(self._excluded)}")
        for r in range(rows):
            row_str = f"  row {r}: "
            for c in range(cols):
                ch_id = r * cols + c
                rc = (r, c)
                if rc in self._excluded:
                    row_str += "[X]"
                elif ch_id in stim:
                    row_str += "[S]"
                elif ch_id in up:
                    row_str += "[U]"
                elif ch_id in down:
                    row_str += "[D]"
                elif ch_id in off:
                    row_str += "[.]"
                else:
                    row_str += "[?]"
            print(row_str)
        print()
