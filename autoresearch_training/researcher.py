"""
researcher.py

TrainingResearcher — main orchestrator for the longitudinal stimulation
parameter search in autoresearch_training/.

Operates on a multi-session timescale (hours to days), in contrast to
autoresearch/researcher.py which evaluates each configuration in a single
~90-second session.  Each regime evaluation spans sessions_per_regime sessions,
interleaved with periodic no-stim control sessions for drift correction.

Usage:
    researcher = TrainingResearcher(
        project_root=".",
        session_seconds=180,
        sessions_per_regime=6,
        min_sessions_before_decision=6,
    )
    researcher.setup()
    researcher.run()
    researcher.summarize()
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from autoresearch_training.constraints import StimConstraintError
from autoresearch_training.stimulation import (
    StimRegime,
    describe_regime,
    save_regime,
    load_regime,
    increase_reward_amplitude,
    decrease_reward_amplitude,
    increase_penalty_amplitude,
    decrease_penalty_amplitude,
    increase_reward_duration,
    decrease_reward_duration,
    increase_penalty_duration,
    decrease_penalty_duration,
    increase_reward_pulses,
    decrease_reward_pulses,
    increase_penalty_pulses,
    decrease_penalty_pulses,
    increase_reward_frequency,
    decrease_reward_frequency,
    swap_reward_penalty_ratio,
)
from autoresearch_training.session_runner import SessionResult, run_session
from autoresearch_training.learning import LearningCurve, compute_learning_curve, subtract_drift, detect_saturation
from autoresearch_training.control import ControlManager
from autoresearch_training.regime_logger import RegimeLogger


# ---------------------------------------------------------------------------
# Perturbation catalogue
# ---------------------------------------------------------------------------

# Tier 1 — Amplitude (highest expected effect size, always active)
_TIER1: list[tuple[str, Callable]] = [
    ("increase_reward_amplitude",  increase_reward_amplitude),
    ("decrease_reward_amplitude",  decrease_reward_amplitude),
    ("increase_penalty_amplitude", increase_penalty_amplitude),
    ("decrease_penalty_amplitude", decrease_penalty_amplitude),
]

# Tier 2 — Duration and pulse count (unlocked after 10 regimes)
_TIER2: list[tuple[str, Callable]] = [
    ("increase_reward_duration",   increase_reward_duration),
    ("decrease_reward_duration",   decrease_reward_duration),
    ("increase_penalty_duration",  increase_penalty_duration),
    ("decrease_penalty_duration",  decrease_penalty_duration),
    ("increase_reward_pulses",     increase_reward_pulses),
    ("decrease_reward_pulses",     decrease_reward_pulses),
    ("increase_penalty_pulses",    increase_penalty_pulses),
    ("decrease_penalty_pulses",    decrease_penalty_pulses),
]

# Tier 3 — Frequency, ISI, and ratio (unlocked after 20 regimes)
_TIER3: list[tuple[str, Callable]] = [
    ("increase_reward_frequency",  increase_reward_frequency),
    ("decrease_reward_frequency",  decrease_reward_frequency),
    ("swap_reward_penalty_ratio",  swap_reward_penalty_ratio),
]


class TrainingResearcher:
    """
    Longitudinal stimulation parameter optimiser.

    Parameters
    ----------
    project_root                : Path to the pyxdaq-fu project root.
    session_seconds             : Duration of each session in seconds (required).
    sessions_per_regime         : Sessions to run per regime before deciding (required).
    min_sessions_before_decision: Minimum sessions before a keep/discard is made (required).
    control_interval            : Run a control session every N total sessions.
    learning_pvalue_threshold   : p-value cutoff for declaring a slope significant.
    log_dir                     : Where TSV logs are written.  Defaults to
                                  project_root/autoresearch_training/logs/.
    mode                        : "rhx" or "mock".
    """

    def __init__(
        self,
        project_root: str,
        session_seconds: int,
        sessions_per_regime: int,
        min_sessions_before_decision: int,
        control_interval: int = 5,
        learning_pvalue_threshold: float = 0.10,
        log_dir: Optional[str] = None,
        mode: str = "rhx",
        config_path: Optional[str] = None,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._session_seconds              = session_seconds
        self._sessions_per_regime          = sessions_per_regime
        self._min_sessions_before_decision = min_sessions_before_decision
        self._control_interval             = control_interval
        self._pvalue_threshold             = learning_pvalue_threshold
        self._mode                         = mode
        self._config_path = config_path or str(self._project_root / "channel_config.json")

        if log_dir is None:
            log_dir = str(self._project_root / "autoresearch_training" / "logs")

        self._logger  = RegimeLogger(log_dir)
        self._control = ControlManager(
            control_interval=control_interval,
            session_seconds=session_seconds,
            min_control_sessions=3,
            mode=mode,
            config_path=self._config_path,
        )

        # State
        self._baseline_regime: Optional[StimRegime] = None
        self._current_regime:  Optional[StimRegime] = None
        self._baseline_curve:  Optional[LearningCurve] = None
        self._total_sessions:  int = 0
        self._regimes_evaluated: int = 0

        # Momentum tracking: record last two keep decisions and their perturbation name
        self._last_kept: list[tuple[str, float]] = []   # [(perturbation_name, dc_slope)]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """
        Initialise the research session:
          1. Verify hardware (RHX or mock)
          2. Load or create the baseline StimRegime
          3. Run min_sessions_before_decision sessions under baseline
          4. Run initial control sessions (at least 3)
          5. Compute and print the baseline LearningCurve
        """
        print("[setup] Verifying hardware ...", file=sys.stderr)
        if self._mode == "rhx":
            self._verify_rhx()

        print("[setup] Loading baseline regime ...", file=sys.stderr)
        self._baseline_regime = self._load_or_create_baseline()
        self._current_regime  = self._baseline_regime

        print("[setup] Running baseline sessions ...", file=sys.stderr)
        baseline_results: list[SessionResult] = []
        for i in range(self._min_sessions_before_decision):
            result = self._run_one_session(self._baseline_regime, session_index=i)
            baseline_results.append(result)
            self._logger.log_session(result, self._baseline_regime)
            self._total_sessions += 1
            # Interleave control sessions on schedule
            if self._control.should_run_control(self._total_sessions):
                ctrl_result = self._control.run_control_session(
                    current_regime=self._baseline_regime
                )
                self._logger.log_control_session(ctrl_result)

        # Ensure at least 3 control sessions before proceeding
        while self._control.n_control_sessions < 3:
            ctrl_result = self._control.run_control_session(
                current_regime=self._baseline_regime
            )
            self._logger.log_control_session(ctrl_result)

        self._baseline_curve = compute_learning_curve(
            baseline_results, self._pvalue_threshold
        )

        drift_rate = self._control.get_drift_rate()
        print(
            f"\n[setup] Baseline complete:\n"
            f"  Regime:             {describe_regime(self._baseline_regime)}\n"
            f"  Baseline slope:     {self._baseline_curve.slope:.4f}/session\n"
            f"  Baseline delta:     {self._baseline_curve.delta:.3f}\n"
            f"  Drift rate (ctrl):  {drift_rate:.4f}/session\n"
            f"  Control sessions:   {self._control.n_control_sessions}\n",
            file=sys.stderr,
        )

    def run(self, max_regimes: Optional[int] = None) -> None:
        """
        Main optimisation loop.

        Each iteration:
          1. Select a perturbation from _select_perturbation()
          2. Evaluate the new regime over sessions_per_regime sessions
          3. Interleave control sessions per control_interval
          4. Compute LearningCurve and apply drift correction
          5. Keep or discard based on drift-corrected delta
          6. Log to regime_logger
          7. Every 20 regimes, call summarize() and append to program.md
        """
        if self._current_regime is None:
            raise RuntimeError("Call setup() before run()")

        regime_count = 0

        try:
            while max_regimes is None or regime_count < max_regimes:
                if not self._check_culture_health():
                    print(
                        "[run] Culture health check failed — stopping loop.",
                        file=sys.stderr,
                    )
                    break

                # Select and apply perturbation
                perturb_name, candidate = self._select_perturbation()
                if candidate is None:
                    print(
                        f"[run] Could not find a valid perturbation after retries — stopping.",
                        file=sys.stderr,
                    )
                    break

                print(
                    f"\n[run] Regime {regime_count + 1}: {perturb_name}\n"
                    f"       {describe_regime(candidate)}",
                    file=sys.stderr,
                )

                raw_curve, dc_curve = self._evaluate_regime(candidate)
                keep = self._decide(self._baseline_curve, dc_curve)

                if keep:
                    status = "keep"
                    self._current_regime = candidate
                    self._baseline_curve = dc_curve
                    self._last_kept.append((perturb_name, dc_curve.slope))
                    if len(self._last_kept) > 2:
                        self._last_kept.pop(0)
                elif dc_curve.n_sessions < self._min_sessions_before_decision:
                    status = "inconclusive"
                    self._last_kept.clear()
                else:
                    status = "discard"
                    self._last_kept.clear()

                dc_slope = dc_curve.slope if dc_curve is not None else 0.0
                description = (
                    f"{perturb_name}; dc_slope={dc_slope:.4f}; "
                    f"dc_delta={dc_curve.delta:.3f}; p={dc_curve.slope_pvalue:.3f}"
                )
                self._logger.log_regime_result(
                    regime=candidate,
                    curve=raw_curve,
                    drift_corrected_curve=dc_curve,
                    status=status,
                    description=description,
                )

                regime_count += 1
                self._regimes_evaluated += 1

                print(
                    f"  → {status.upper()}: dc_slope={dc_slope:.4f}, "
                    f"dc_delta={dc_curve.delta:.3f}, p={dc_curve.slope_pvalue:.3f}",
                    file=sys.stderr,
                )

                # Periodic summary
                if self._regimes_evaluated % 20 == 0:
                    summary = self.summarize()
                    self._logger.write_summary(summary)
                    self._append_to_program_md(summary)

        except KeyboardInterrupt:
            print("\n[run] Interrupted by user.", file=sys.stderr)

    def summarize(self) -> str:
        """
        Return a synthesis paragraph covering search findings so far.
        """
        all_rows = self._logger.get_all_regimes()
        kept     = [r for r in all_rows if r.get("status") == "keep"]
        discarded = [r for r in all_rows if r.get("status") == "discard"]
        inconclusive = [r for r in all_rows if r.get("status") == "inconclusive"]

        best = self._logger.get_best_regime()
        best_str = f"regime {best[0]} (dc_slope={best[1]:.4f})" if best else "none yet"

        ts = datetime.now(timezone.utc).isoformat()

        # Identify which perturbation types correlated with keeps
        keep_perturbations = [r.get("description", "").split(";")[0].strip() for r in kept]
        pert_counts: dict[str, int] = {}
        for p in keep_perturbations:
            pert_counts[p] = pert_counts.get(p, 0) + 1
        top_perts = sorted(pert_counts.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{p} ({n}x)" for p, n in top_perts) if top_perts else "none"

        return (
            f"### autoresearch_training summary — {ts}\n\n"
            f"Regimes evaluated: {self._regimes_evaluated} "
            f"(kept: {len(kept)}, discarded: {len(discarded)}, "
            f"inconclusive: {len(inconclusive)})\n\n"
            f"Best regime so far: {best_str}\n\n"
            f"Most productive perturbations: {top_str}\n\n"
            f"Drift rate (current): {self._control.get_drift_rate()}\n\n"
            f"Control sessions run: {self._control.n_control_sessions}\n"
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _evaluate_regime(
        self, regime: StimRegime
    ) -> tuple[LearningCurve, LearningCurve]:
        """
        Run sessions_per_regime sessions, interleave control sessions,
        compute raw and drift-corrected LearningCurves.

        Returns (raw_curve, drift_corrected_curve).
        If drift correction is not yet available, drift_corrected_curve
        equals raw_curve (no correction applied).
        """
        results: list[SessionResult] = []

        for i in range(self._sessions_per_regime):
            result = self._run_one_session(regime, session_index=i)
            results.append(result)
            self._logger.log_session(result, regime)
            self._total_sessions += 1

            if self._control.should_run_control(self._total_sessions):
                ctrl_result = self._control.run_control_session(
                    session_index=self._control.n_control_sessions,
                    current_regime=regime,
                )
                self._logger.log_control_session(ctrl_result)

        raw_curve = compute_learning_curve(results, self._pvalue_threshold)

        drift_curve = self._control.get_drift_curve()
        if drift_curve is not None:
            dc_curve = subtract_drift(raw_curve, drift_curve)
        else:
            dc_curve = raw_curve
            print(
                "[evaluate] Drift correction not yet available "
                f"(need {3} control sessions, have {self._control.n_control_sessions}). "
                "Using raw curve.",
                file=sys.stderr,
            )

        return raw_curve, dc_curve

    def _select_perturbation(self) -> tuple[str, Optional[StimRegime]]:
        """
        Select the next perturbation to try.

        Strategy:
          - Phase 1 (0–9 regimes): amplitude perturbations only (Tier 1)
          - Phase 2 (10–19 regimes): amplitude + duration/pulses (Tier 1+2)
          - Phase 3 (20+ regimes): all perturbations (Tier 1+2+3)
          - Momentum: if the same direction improved twice in a row, repeat it (60% chance)
          - Exploration: 10% chance of a larger jump (40% instead of default 20%)
          - Never return a regime that violates constraints
        """
        available: list[tuple[str, Callable]] = list(_TIER1)
        if self._regimes_evaluated >= 10:
            available += _TIER2
        if self._regimes_evaluated >= 20:
            available += _TIER3

        # Momentum: prefer the direction that succeeded recently
        if (
            len(self._last_kept) == 2
            and self._last_kept[0][0] == self._last_kept[1][0]
            and random.random() < 0.60
        ):
            momentum_name = self._last_kept[0][0]
            momentum_fns  = [fn for name, fn in available if name == momentum_name]
            if momentum_fns:
                available = [(momentum_name, momentum_fns[0])]

        # Shuffle to avoid always picking the first entry
        random.shuffle(available)

        large_jump = random.random() < 0.10   # 10% exploration rate

        for name, fn in available:
            try:
                if large_jump and "amplitude" in name:
                    candidate = fn(self._current_regime, pct=0.40)
                elif large_jump and "duration" in name:
                    candidate = fn(self._current_regime, pct=0.40)
                else:
                    candidate = fn(self._current_regime)
                return name, candidate
            except StimConstraintError:
                continue

        return "none", None

    def _decide(
        self,
        current_curve: Optional[LearningCurve],
        candidate_curve: LearningCurve,
    ) -> bool:
        """
        Keep if:
          - drift_corrected delta > 0
          - slope p-value < learning_pvalue_threshold
          - n_sessions >= min_sessions_before_decision

        Inconclusive if p-value is borderline (0.10 <= p < 0.20).
        """
        if candidate_curve.n_sessions < self._min_sessions_before_decision:
            return False

        if candidate_curve.delta <= 0:
            return False

        if candidate_curve.slope_pvalue >= self._pvalue_threshold:
            return False

        # Optional: require the candidate to be better than the current regime
        if current_curve is not None and candidate_curve.slope <= current_curve.slope:
            return False

        return True

    def _check_culture_health(self) -> bool:
        """
        Return False if median peak_firing_hz over the last 3 control sessions
        has dropped below 5 Hz — a sign of culture degradation.
        """
        recent = self._control.all_control_results[-3:]
        if len(recent) < 3:
            return True  # not enough data yet — assume healthy
        median_hz = sorted(r.peak_firing_hz for r in recent)[len(recent) // 2]
        if median_hz < 5.0:
            print(
                f"[health] WARNING: median peak_firing_hz over last 3 control sessions "
                f"= {median_hz:.2f} Hz < 5.0 Hz — culture may be degrading.",
                file=sys.stderr,
            )
            return False
        return True

    def _run_one_session(
        self, regime: StimRegime, session_index: int
    ) -> SessionResult:
        return run_session(
            regime=regime,
            session_seconds=self._session_seconds,
            session_index=session_index,
            mode=self._mode,
            config_path=self._config_path,
        )

    def _load_or_create_baseline(self) -> StimRegime:
        """
        Load saved baseline regime if present, otherwise create a default
        from stimulator.py's default parameter values.

        The default baseline matches the default StimConfig values in stimulator.py:
            reward_amplitude: 10 µA (treated as 10 µV-equivalent in search space)
            reward_duration:  200 µs = 0.2 ms
            penalty_amplitude: 20 µA → 20 µV-equivalent
            penalty_duration:  400 µs = 0.4 ms
        """
        baseline_path = (
            self._project_root / "autoresearch_training" / "baseline_regime.json"
        )
        if baseline_path.exists():
            regime = load_regime(str(baseline_path))
            print(f"[setup] Loaded baseline from {baseline_path}", file=sys.stderr)
            return regime

        regime = StimRegime(
            reward_amplitude_uv=10.0,
            reward_duration_ms=0.2,
            reward_frequency_hz=10.0,
            reward_n_pulses=1,
            reward_isi_ms=10.0,
            penalty_amplitude_uv=20.0,
            penalty_duration_ms=0.4,
            penalty_frequency_hz=10.0,
            penalty_n_pulses=1,
            penalty_isi_ms=10.0,
            reward_penalty_ratio=10.0 / 20.0,
            notes="Default baseline from stimulator.py defaults",
        )
        save_regime(regime, str(baseline_path))
        print(
            f"[setup] Created default baseline regime → {baseline_path}",
            file=sys.stderr,
        )
        return regime

    def _verify_rhx(self) -> None:
        """Attempt a socket connection to RHX to verify hardware is live."""
        import socket as _socket
        try:
            s = _socket.create_connection(("127.0.0.1", 5000), timeout=5.0)
            s.close()
        except OSError as exc:
            raise RuntimeError(
                f"[setup] Cannot reach RHX at 127.0.0.1:5000 — is RHX running? ({exc})"
            ) from exc

    def _append_to_program_md(self, summary: str) -> None:
        """Append a summary block to program.md under Session History & Findings."""
        program_path = self._project_root / "autoresearch_training" / "program.md"
        if not program_path.exists():
            return
        with open(program_path, "a") as f:
            f.write(f"\n{summary}\n")
