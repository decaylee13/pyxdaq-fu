"""
stimulation.py

StimRegime dataclass and perturbation functions for autoresearch_training/.

StimRegime is the mutable search space for this library — the equivalent of
channel_config.json in autoresearch/, but for stimulation feedback parameters
rather than electrode layout.  Each regime describes a complete reward/penalty
pulse waveform delivered in response to game events.

All perturbation functions:
  - Return a new StimRegime (never mutate the input)
  - Auto-generate a new regime_id from the updated parameter values
  - Call constraints.validate_regime() before returning; raise StimConstraintError if violated
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from autoresearch_training.constraints import validate_regime, StimConstraintError  # noqa: F401


@dataclass
class StimRegime:
    # Reward pulse (delivered on paddle hit)
    reward_amplitude_uv: float       # e.g. 100–800 µV
    reward_duration_ms: float        # e.g. 0.1–2.0 ms
    reward_frequency_hz: float       # pulse repetition rate in burst, e.g. 1–100 Hz
    reward_n_pulses: int             # number of pulses per reward event, e.g. 1–5
    reward_isi_ms: float             # inter-stimulus interval within burst, ms

    # Penalty pulse (delivered on miss)
    penalty_amplitude_uv: float
    penalty_duration_ms: float
    penalty_frequency_hz: float
    penalty_n_pulses: int
    penalty_isi_ms: float

    # Regime-level parameters
    reward_penalty_ratio: float      # reward_amplitude_uv / penalty_amplitude_uv (derived, for logging)
    stim_enabled: bool = True        # False = control condition (no stim)

    # Metadata
    regime_id: str = ""              # 7-char hash, auto-generated
    created_at: str = ""             # ISO timestamp
    notes: str = ""                  # human or agent annotation

    def __post_init__(self) -> None:
        if not self.regime_id:
            self.regime_id = generate_regime_id(self)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ID and description
# ---------------------------------------------------------------------------

def generate_regime_id(regime: StimRegime) -> str:
    """Return a 7-char SHA-256 hash of the numeric parameter values."""
    params = {
        "reward_amplitude_uv":  regime.reward_amplitude_uv,
        "reward_duration_ms":   regime.reward_duration_ms,
        "reward_frequency_hz":  regime.reward_frequency_hz,
        "reward_n_pulses":      regime.reward_n_pulses,
        "reward_isi_ms":        regime.reward_isi_ms,
        "penalty_amplitude_uv": regime.penalty_amplitude_uv,
        "penalty_duration_ms":  regime.penalty_duration_ms,
        "penalty_frequency_hz": regime.penalty_frequency_hz,
        "penalty_n_pulses":     regime.penalty_n_pulses,
        "penalty_isi_ms":       regime.penalty_isi_ms,
        "stim_enabled":         regime.stim_enabled,
    }
    digest = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()
    return digest[:7]


def describe_regime(regime: StimRegime) -> str:
    """One-line human-readable summary for logging."""
    if not regime.stim_enabled:
        return "CONTROL (stim disabled)"
    return (
        f"reward={regime.reward_amplitude_uv:.0f}µV×{regime.reward_duration_ms:.2f}ms"
        f"×{regime.reward_n_pulses}p "
        f"penalty={regime.penalty_amplitude_uv:.0f}µV×{regime.penalty_duration_ms:.2f}ms"
        f"×{regime.penalty_n_pulses}p "
        f"ratio={regime.reward_penalty_ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def regime_to_dict(regime: StimRegime) -> dict:
    return asdict(regime)


def regime_from_dict(d: dict) -> StimRegime:
    return StimRegime(**d)


def save_regime(regime: StimRegime, path: str) -> None:
    """Atomic JSON write — safe against partial-write corruption."""
    data = json.dumps(regime_to_dict(regime), indent=2)
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_regime(path: str) -> StimRegime:
    with open(path) as f:
        return regime_from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Internal copy helper
# ---------------------------------------------------------------------------

def _copy_with(regime: StimRegime, **kwargs) -> StimRegime:
    """
    Return a validated copy of regime with the given fields overridden.
    Recomputes reward_penalty_ratio and generates a new regime_id automatically.
    """
    d = regime_to_dict(regime)
    d.update(kwargs)
    # Recompute derived ratio
    d["reward_penalty_ratio"] = d["reward_amplitude_uv"] / d["penalty_amplitude_uv"]
    # Clear metadata so fresh id/timestamp are generated
    d["regime_id"] = ""
    d["created_at"] = ""
    new_regime = regime_from_dict(d)
    validate_regime(new_regime)
    return new_regime


# ---------------------------------------------------------------------------
# Perturbation functions — each returns a new StimRegime, never mutates
# ---------------------------------------------------------------------------

def increase_reward_amplitude(regime: StimRegime, pct: float = 0.20) -> StimRegime:
    return _copy_with(regime, reward_amplitude_uv=regime.reward_amplitude_uv * (1 + pct))


def decrease_reward_amplitude(regime: StimRegime, pct: float = 0.20) -> StimRegime:
    return _copy_with(regime, reward_amplitude_uv=regime.reward_amplitude_uv * (1 - pct))


def increase_penalty_amplitude(regime: StimRegime, pct: float = 0.20) -> StimRegime:
    return _copy_with(regime, penalty_amplitude_uv=regime.penalty_amplitude_uv * (1 + pct))


def decrease_penalty_amplitude(regime: StimRegime, pct: float = 0.20) -> StimRegime:
    return _copy_with(regime, penalty_amplitude_uv=regime.penalty_amplitude_uv * (1 - pct))


def increase_reward_duration(regime: StimRegime, pct: float = 0.20) -> StimRegime:
    return _copy_with(regime, reward_duration_ms=regime.reward_duration_ms * (1 + pct))


def decrease_reward_duration(regime: StimRegime, pct: float = 0.20) -> StimRegime:
    return _copy_with(regime, reward_duration_ms=regime.reward_duration_ms * (1 - pct))


def increase_penalty_duration(regime: StimRegime, pct: float = 0.20) -> StimRegime:
    return _copy_with(regime, penalty_duration_ms=regime.penalty_duration_ms * (1 + pct))


def decrease_penalty_duration(regime: StimRegime, pct: float = 0.20) -> StimRegime:
    return _copy_with(regime, penalty_duration_ms=regime.penalty_duration_ms * (1 - pct))


def increase_reward_pulses(regime: StimRegime) -> StimRegime:
    """Add one pulse to the reward burst (+1, up to MAX_N_PULSES)."""
    return _copy_with(regime, reward_n_pulses=regime.reward_n_pulses + 1)


def decrease_reward_pulses(regime: StimRegime) -> StimRegime:
    """Remove one pulse from the reward burst (-1, minimum 1)."""
    return _copy_with(regime, reward_n_pulses=max(1, regime.reward_n_pulses - 1))


def increase_penalty_pulses(regime: StimRegime) -> StimRegime:
    return _copy_with(regime, penalty_n_pulses=regime.penalty_n_pulses + 1)


def decrease_penalty_pulses(regime: StimRegime) -> StimRegime:
    return _copy_with(regime, penalty_n_pulses=max(1, regime.penalty_n_pulses - 1))


def increase_reward_frequency(regime: StimRegime, pct: float = 0.25) -> StimRegime:
    return _copy_with(regime, reward_frequency_hz=regime.reward_frequency_hz * (1 + pct))


def decrease_reward_frequency(regime: StimRegime, pct: float = 0.25) -> StimRegime:
    return _copy_with(regime, reward_frequency_hz=regime.reward_frequency_hz * (1 - pct))


def swap_reward_penalty_ratio(regime: StimRegime, delta: float = 0.10) -> StimRegime:
    """
    Shift intensity balance between reward and penalty.
    delta > 0 → stronger reward relative to penalty (increase reward, decrease penalty).
    delta < 0 → stronger penalty relative to reward.
    Each amplitude moves by delta * 100% in opposite directions.
    """
    new_reward  = regime.reward_amplitude_uv  * (1 + delta)
    new_penalty = regime.penalty_amplitude_uv * (1 - delta)
    return _copy_with(regime, reward_amplitude_uv=new_reward, penalty_amplitude_uv=new_penalty)


def disable_stim(regime: StimRegime) -> StimRegime:
    """
    Return a copy of this regime with stim_enabled=False.
    Used to generate the control condition (no-stim baseline) for drift correction.
    All pulse parameters are preserved so the regime can be re-enabled later.
    """
    d = regime_to_dict(regime)
    d["stim_enabled"] = False
    d["reward_penalty_ratio"] = d["reward_amplitude_uv"] / d["penalty_amplitude_uv"]
    d["regime_id"] = ""
    d["created_at"] = ""
    new_regime = regime_from_dict(d)
    validate_regime(new_regime)
    return new_regime
