"""
constraints.py

Hard safety limits on stimulation parameters for autoresearch_training/.

All regime perturbations pass through validate_regime() before being applied to hardware.
The charge density check is the most safety-critical: it prevents electrode damage by
capping the total energy delivered per stimulation event.

Unit note:
    StimRegime uses µV for amplitude and ms for duration as search-space coordinates.
    When applied to RHX hardware (stimulator.py), the numeric value of amplitude_uv
    is passed directly as µA (same magnitude, different unit) and duration_ms * 1000
    is passed as µs.  Physical calibration of the µV→µA mapping should be done using
    measured electrode impedance before running on a live culture.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Safety bounds
# ---------------------------------------------------------------------------

MAX_REWARD_AMPLITUDE_UV: float  = 800.0
MIN_REWARD_AMPLITUDE_UV: float  = 50.0
MAX_PENALTY_AMPLITUDE_UV: float = 1200.0
MIN_PENALTY_AMPLITUDE_UV: float = 50.0
MAX_DURATION_MS: float          = 3.0
MIN_DURATION_MS: float          = 0.05
MAX_FREQUENCY_HZ: float         = 200.0
MIN_FREQUENCY_HZ: float         = 1.0
MAX_N_PULSES: int               = 10
MIN_N_PULSES: int               = 1

# Charge density safety ceiling.
# Proxy formula: amplitude_uv * duration_ms * n_pulses / 1e6
# (scales µV·ms·count to a nC-equivalent assuming 1 MΩ electrode impedance).
# Recalibrate this constant based on measured headstage impedance before use.
# At 1 MΩ: ceiling = 5.0 nC-equiv → blocks amplitude*duration*n_pulses > 5e6.
# At 100 kΩ: divide MAX by 10 for equivalent physical charge protection.
MAX_TOTAL_CHARGE_PER_EVENT_NC: float = 5.0   # nC-equivalent (see formula in check_charge_density)


class StimConstraintError(Exception):
    """Raised when a StimRegime violates a hard safety constraint."""
    pass


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_amplitude_bounds(regime: "StimRegime") -> None:
    if not (MIN_REWARD_AMPLITUDE_UV <= regime.reward_amplitude_uv <= MAX_REWARD_AMPLITUDE_UV):
        raise StimConstraintError(
            f"reward_amplitude_uv {regime.reward_amplitude_uv:.1f} outside "
            f"[{MIN_REWARD_AMPLITUDE_UV}, {MAX_REWARD_AMPLITUDE_UV}]"
        )
    if not (MIN_PENALTY_AMPLITUDE_UV <= regime.penalty_amplitude_uv <= MAX_PENALTY_AMPLITUDE_UV):
        raise StimConstraintError(
            f"penalty_amplitude_uv {regime.penalty_amplitude_uv:.1f} outside "
            f"[{MIN_PENALTY_AMPLITUDE_UV}, {MAX_PENALTY_AMPLITUDE_UV}]"
        )


def check_duration_bounds(regime: "StimRegime") -> None:
    for attr in ("reward_duration_ms", "penalty_duration_ms"):
        val = getattr(regime, attr)
        if not (MIN_DURATION_MS <= val <= MAX_DURATION_MS):
            raise StimConstraintError(
                f"{attr} {val:.3f} outside [{MIN_DURATION_MS}, {MAX_DURATION_MS}]"
            )


def check_frequency_bounds(regime: "StimRegime") -> None:
    for attr in ("reward_frequency_hz", "penalty_frequency_hz"):
        val = getattr(regime, attr)
        if not (MIN_FREQUENCY_HZ <= val <= MAX_FREQUENCY_HZ):
            raise StimConstraintError(
                f"{attr} {val:.1f} outside [{MIN_FREQUENCY_HZ}, {MAX_FREQUENCY_HZ}]"
            )


def check_pulse_counts(regime: "StimRegime") -> None:
    for attr in ("reward_n_pulses", "penalty_n_pulses"):
        val = getattr(regime, attr)
        if not (MIN_N_PULSES <= val <= MAX_N_PULSES):
            raise StimConstraintError(
                f"{attr} {val} outside [{MIN_N_PULSES}, {MAX_N_PULSES}]"
            )


def check_charge_density(regime: "StimRegime") -> None:
    """
    Check per-event charge proxy for both reward and penalty pulses.

    Formula: (amplitude_uv * duration_ms * n_pulses) / 1e6
    This scales µV·ms·count to a nC-equivalent under the 1 MΩ electrode assumption.
    Exceeding MAX_TOTAL_CHARGE_PER_EVENT_NC risks electrolytic damage to the culture.
    """
    _SCALE = 1e6

    reward_proxy = (
        regime.reward_amplitude_uv
        * regime.reward_duration_ms
        * regime.reward_n_pulses
    ) / _SCALE

    penalty_proxy = (
        regime.penalty_amplitude_uv
        * regime.penalty_duration_ms
        * regime.penalty_n_pulses
    ) / _SCALE

    if reward_proxy > MAX_TOTAL_CHARGE_PER_EVENT_NC:
        raise StimConstraintError(
            f"Reward charge proxy {reward_proxy:.2e} nC-equiv exceeds ceiling "
            f"{MAX_TOTAL_CHARGE_PER_EVENT_NC}. "
            f"Recalibrate MAX_TOTAL_CHARGE_PER_EVENT_NC for actual electrode impedance."
        )
    if penalty_proxy > MAX_TOTAL_CHARGE_PER_EVENT_NC:
        raise StimConstraintError(
            f"Penalty charge proxy {penalty_proxy:.2e} nC-equiv exceeds ceiling "
            f"{MAX_TOTAL_CHARGE_PER_EVENT_NC}. "
            f"Recalibrate MAX_TOTAL_CHARGE_PER_EVENT_NC for actual electrode impedance."
        )


# ---------------------------------------------------------------------------
# Composite validator
# ---------------------------------------------------------------------------

def validate_regime(regime: "StimRegime") -> None:
    """Run all safety checks. Raises StimConstraintError on first violation."""
    check_amplitude_bounds(regime)
    check_duration_bounds(regime)
    check_frequency_bounds(regime)
    check_pulse_counts(regime)
    check_charge_density(regime)
