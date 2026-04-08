"""
perturbations.py

All valid config perturbation functions for the autoresearch loop.
Each function takes a config dict (and optionally a decoder params dict),
returns a NEW dict (never mutates in place), and raises
ConstraintViolationError if the result would violate hard constraints.

Channels are identified by (row, col) grid coordinates in the public API
and converted to flat integer IDs internally.
"""

from __future__ import annotations

import copy

from autoresearch.constraints import (
    MIN_STIM_REC_SEPARATION,
    ConstraintViolationError,
    _chebyshev,
    _id_to_rc,
    validate_config,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rc_to_id(row: int, col: int, grid_cols: int) -> int:
    """Convert (row, col) to flat channel ID."""
    return row * grid_cols + col


def _deep(config: dict) -> dict:
    """Deep-copy a config dict (avoids accidental mutation)."""
    return copy.deepcopy(config)


def _sync_recording(config: dict) -> dict:
    """
    Ensure recording_channels equals up_channels ∪ down_channels.
    Call this after any role reassignment.
    """
    up = set(config.get("up_channels", []))
    down = set(config.get("down_channels", []))
    config["recording_channels"] = sorted(up | down)
    return config


# ---------------------------------------------------------------------------
# Tier 1 — Channel role swaps (lowest risk)
# ---------------------------------------------------------------------------

def swap_region(config: dict, channel: tuple[int, int]) -> dict:
    """
    Flip a single channel between the UP and DOWN recording regions.
    The channel must currently be in either UP or DOWN.
    Raises ConstraintViolationError if the move would drop a region
    below its minimum.
    """
    cols = config.get("grid_cols", 4)
    ch_id = _rc_to_id(*channel, cols)

    up = config.get("up_channels", [])
    down = config.get("down_channels", [])

    if ch_id not in up and ch_id not in down:
        raise ConstraintViolationError(
            f"Channel {channel} (id={ch_id}) is not in UP or DOWN region."
        )

    cfg = _deep(config)
    if ch_id in cfg["up_channels"]:
        cfg["up_channels"].remove(ch_id)
        cfg["down_channels"].append(ch_id)
        cfg["down_channels"].sort()
    else:
        cfg["down_channels"].remove(ch_id)
        cfg["up_channels"].append(ch_id)
        cfg["up_channels"].sort()

    _sync_recording(cfg)
    validate_config(cfg)
    return cfg


def silence_channel(config: dict, channel: tuple[int, int]) -> dict:
    """
    Move a recording (UP or DOWN) channel to the disabled (OFF) list.
    Raises ConstraintViolationError if removing it would violate region
    minimums.
    """
    cols = config.get("grid_cols", 4)
    ch_id = _rc_to_id(*channel, cols)

    up = config.get("up_channels", [])
    down = config.get("down_channels", [])

    if ch_id not in up and ch_id not in down:
        raise ConstraintViolationError(
            f"Channel {channel} (id={ch_id}) is not a recording channel."
        )

    cfg = _deep(config)
    if ch_id in cfg["up_channels"]:
        cfg["up_channels"].remove(ch_id)
    else:
        cfg["down_channels"].remove(ch_id)

    cfg.setdefault("disabled_channels", [])
    if ch_id not in cfg["disabled_channels"]:
        cfg["disabled_channels"].append(ch_id)
        cfg["disabled_channels"].sort()

    _sync_recording(cfg)
    validate_config(cfg)
    return cfg


def activate_channel(config: dict, channel: tuple[int, int]) -> dict:
    """
    Move a disabled (OFF) channel into the recording pool, assigning it to
    whichever region (UP or DOWN) currently has fewer channels.
    Raises ConstraintViolationError if the channel is not currently OFF.
    """
    cols = config.get("grid_cols", 4)
    ch_id = _rc_to_id(*channel, cols)

    disabled = config.get("disabled_channels", [])
    if ch_id not in disabled:
        raise ConstraintViolationError(
            f"Channel {channel} (id={ch_id}) is not in the disabled list."
        )

    cfg = _deep(config)
    cfg["disabled_channels"].remove(ch_id)

    # Assign to smaller region
    if len(cfg.get("up_channels", [])) <= len(cfg.get("down_channels", [])):
        cfg.setdefault("up_channels", [])
        cfg["up_channels"].append(ch_id)
        cfg["up_channels"].sort()
    else:
        cfg.setdefault("down_channels", [])
        cfg["down_channels"].append(ch_id)
        cfg["down_channels"].sort()

    _sync_recording(cfg)
    validate_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Tier 2 — Stimulation reassignment (medium risk)
# ---------------------------------------------------------------------------

def swap_stim_rec(
    config: dict,
    stim_ch: tuple[int, int],
    rec_ch: tuple[int, int],
) -> dict:
    """
    Swap roles between a STIM channel and a REC (UP/DOWN) channel.
    The new STIM position must be within MIN_STIM_REC_SEPARATION+1 grid
    steps of the old STIM position (keeps the stimulation zone close).
    Raises ConstraintViolationError if the swap is invalid.
    """
    cols = config.get("grid_cols", 4)
    stim_id = _rc_to_id(*stim_ch, cols)
    rec_id = _rc_to_id(*rec_ch, cols)

    if stim_id not in config.get("stim_channels", []):
        raise ConstraintViolationError(
            f"Channel {stim_ch} (id={stim_id}) is not a STIM channel."
        )
    up = config.get("up_channels", [])
    down = config.get("down_channels", [])
    if rec_id not in up and rec_id not in down:
        raise ConstraintViolationError(
            f"Channel {rec_ch} (id={rec_id}) is not a REC (UP/DOWN) channel."
        )

    # The new STIM (rec_ch) must be close to the old STIM (stim_ch)
    if _chebyshev(stim_ch, rec_ch) > MIN_STIM_REC_SEPARATION + 1:
        raise ConstraintViolationError(
            f"New STIM position {rec_ch} is {_chebyshev(stim_ch, rec_ch)} steps "
            f"from old STIM {stim_ch} — maximum allowed drift is "
            f"{MIN_STIM_REC_SEPARATION + 1}."
        )

    cfg = _deep(config)

    # Move old STIM to the same region as rec_ch is leaving from
    if rec_id in cfg["up_channels"]:
        cfg["up_channels"].remove(rec_id)
        cfg["up_channels"].append(stim_id)
        cfg["up_channels"].sort()
    else:
        cfg["down_channels"].remove(rec_id)
        cfg["down_channels"].append(stim_id)
        cfg["down_channels"].sort()

    cfg["stim_channels"].remove(stim_id)
    cfg["stim_channels"].append(rec_id)
    cfg["stim_channels"].sort()

    _sync_recording(cfg)
    validate_config(cfg)
    return cfg


def add_stim_channel(config: dict, channel: tuple[int, int]) -> dict:
    """
    Promote a REC channel to a STIM channel.
    The new STIM must be at least MIN_STIM_REC_SEPARATION grid steps from
    all existing STIM channels (prevents clustered stimulation artefacts).
    Raises ConstraintViolationError if constraints would be violated.
    """
    cols = config.get("grid_cols", 4)
    ch_id = _rc_to_id(*channel, cols)

    up = config.get("up_channels", [])
    down = config.get("down_channels", [])
    if ch_id not in up and ch_id not in down:
        raise ConstraintViolationError(
            f"Channel {channel} (id={ch_id}) must be a REC channel to be "
            "promoted to STIM."
        )

    # Check separation from existing STIM channels
    for s in config.get("stim_channels", []):
        s_rc = _id_to_rc(s, cols)
        dist = _chebyshev(channel, s_rc)
        if dist < MIN_STIM_REC_SEPARATION:
            raise ConstraintViolationError(
                f"New STIM {channel} would be {dist} step(s) from existing STIM "
                f"{s_rc} (minimum: {MIN_STIM_REC_SEPARATION})."
            )

    cfg = _deep(config)
    if ch_id in cfg.get("up_channels", []):
        cfg["up_channels"].remove(ch_id)
    else:
        cfg["down_channels"].remove(ch_id)

    cfg.setdefault("stim_channels", [])
    cfg["stim_channels"].append(ch_id)
    cfg["stim_channels"].sort()

    _sync_recording(cfg)
    validate_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Tier 3 — Decoder parameter tuning
# ---------------------------------------------------------------------------

def adjust_dominance_threshold(params: dict, delta_pct: float) -> dict:
    """
    Adjust the action-decoder dominance threshold by delta_pct.
    e.g. delta_pct=+0.10 → +10% of current value.
    params must contain key "threshold" (float, Hz).
    Raises ValueError if result would be <= 0.
    """
    new_params = dict(params)
    current = float(params["threshold"])
    new_val = current * (1.0 + delta_pct)
    if new_val <= 0:
        raise ValueError(
            f"threshold must be > 0 after adjustment; got {new_val:.4f}"
        )
    new_params["threshold"] = round(new_val, 4)
    return new_params


def adjust_hold_deadband(params: dict, delta_hz: float) -> dict:
    """
    Adjust the HOLD dead-band (dominance threshold) by an absolute delta in Hz.
    Equivalent to adjust_dominance_threshold but uses an absolute offset
    rather than a percentage — useful for fine-grained tuning.
    Raises ValueError if result would be <= 0.
    """
    new_params = dict(params)
    current = float(params["threshold"])
    new_val = current + delta_hz
    if new_val <= 0:
        raise ValueError(
            f"threshold must be > 0 after adjustment; got {new_val:.4f}"
        )
    new_params["threshold"] = round(new_val, 4)
    return new_params


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def describe_perturbation(before: dict, after: dict) -> str:
    """
    Return a human-readable one-line description of what changed between
    two config dicts.  Used for TSV log entries.
    """
    parts: list[str] = []

    roles = {
        "up_channels":       "UP",
        "down_channels":     "DOWN",
        "stim_channels":     "STIM",
        "disabled_channels": "OFF",
    }

    for key, label in roles.items():
        b_set = set(before.get(key, []))
        a_set = set(after.get(key, []))
        added = sorted(a_set - b_set)
        removed = sorted(b_set - a_set)
        if added:
            parts.append(f"+{label}:{added}")
        if removed:
            parts.append(f"-{label}:{removed}")

    # Decoder params diff
    for pkey in ("threshold", "smooth_windows"):
        b_val = before.get(pkey)
        a_val = after.get(pkey)
        if b_val is not None and a_val is not None and b_val != a_val:
            parts.append(f"{pkey}:{b_val}->{a_val}")

    return "; ".join(parts) if parts else "no change"
