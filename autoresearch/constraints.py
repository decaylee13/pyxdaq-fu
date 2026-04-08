"""
constraints.py

Hard constraint validators for channel_config.json.  Every function raises
ConstraintViolationError with a descriptive message on the first violation
found.  validate_config() runs all checks in order and is the single
gatekeeping call used by ConfigManager.save().
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

MIN_UP_CHANNELS: int = 4
MIN_DOWN_CHANNELS: int = 4
MAX_STIM_CHANNELS: int = 4
MIN_STIM_CHANNELS: int = 1
MIN_STIM_REC_SEPARATION: int = 2   # Chebyshev grid steps


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ConstraintViolationError(Exception):
    """Raised when a config dict violates a hard constraint."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _id_to_rc(ch_id: int, grid_cols: int) -> tuple[int, int]:
    """Convert a flat channel ID to a (row, col) grid coordinate."""
    return (ch_id // grid_cols, ch_id % grid_cols)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Chebyshev (L-inf) distance between two grid coordinates."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_region_minimums(config: dict) -> None:
    """Ensure UP and DOWN regions each have the minimum number of channels."""
    up = config.get("up_channels", [])
    down = config.get("down_channels", [])
    if len(up) < MIN_UP_CHANNELS:
        raise ConstraintViolationError(
            f"UP region needs at least {MIN_UP_CHANNELS} channels, got {len(up)}."
        )
    if len(down) < MIN_DOWN_CHANNELS:
        raise ConstraintViolationError(
            f"DOWN region needs at least {MIN_DOWN_CHANNELS} channels, got {len(down)}."
        )


def check_stim_count(config: dict) -> None:
    """Enforce MIN_STIM_CHANNELS <= len(stim_channels) <= MAX_STIM_CHANNELS."""
    stim = config.get("stim_channels", [])
    if len(stim) < MIN_STIM_CHANNELS:
        raise ConstraintViolationError(
            f"At least {MIN_STIM_CHANNELS} STIM channel(s) required, got {len(stim)}."
        )
    if len(stim) > MAX_STIM_CHANNELS:
        raise ConstraintViolationError(
            f"At most {MAX_STIM_CHANNELS} STIM channels allowed, got {len(stim)}."
        )


def check_stim_rec_adjacency(config: dict) -> None:
    """
    Ensure no STIM channel is within MIN_STIM_REC_SEPARATION Chebyshev steps
    of any recording (REC) channel.  This prevents stimulation artefacts from
    contaminating the closest spike-recording sites.
    """
    cols = config.get("grid_cols", 4)
    stim_ids = config.get("stim_channels", [])
    rec_ids = config.get("recording_channels", [])

    for s in stim_ids:
        s_rc = _id_to_rc(s, cols)
        for r in rec_ids:
            r_rc = _id_to_rc(r, cols)
            dist = _chebyshev(s_rc, r_rc)
            if dist < MIN_STIM_REC_SEPARATION:
                raise ConstraintViolationError(
                    f"STIM channel {s} {s_rc} is only {dist} grid step(s) from "
                    f"REC channel {r} {r_rc} — minimum separation is "
                    f"{MIN_STIM_REC_SEPARATION} (Chebyshev)."
                )


# ---------------------------------------------------------------------------
# Aggregate validator
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> None:
    """
    Run all hard constraint checks in order.  Raises ConstraintViolationError
    on the first violation.  Call this before writing any config to disk.
    """
    check_region_minimums(config)
    check_stim_count(config)
    check_stim_rec_adjacency(config)
