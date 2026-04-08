"""
config_manager.py

Single point of access for reading and writing channel_config.json.  All
config mutations in the autoresearch library must go through ConfigManager
so that atomic writes and constraint validation are always enforced.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from autoresearch.constraints import (
    ConstraintViolationError,
    _chebyshev,
    _id_to_rc,
    validate_config,
)


class ConfigManager:
    """
    Handles all reads and writes to a channel_config.json file.

    Atomic writes are implemented via write-to-temp-then-rename so that an
    interrupted save never leaves a half-written config on disk.
    """

    def __init__(self, config_path: str) -> None:
        self._path = Path(config_path)

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """
        Load and validate the config.  Raises ConstraintViolationError if
        the stored config violates hard constraints (e.g. too few UP channels).
        Raises FileNotFoundError if the file does not exist.
        """
        raw = json.loads(self._path.read_text())
        validate_config(raw)
        return raw

    def load_raw(self) -> dict:
        """
        Load the config without constraint validation.  Used during setup()
        to inspect the initial state before it has been made autoresearch-valid.
        """
        return json.loads(self._path.read_text())

    def save(self, config: dict) -> None:
        """
        Validate then atomically write config to disk.
        Writes to a .tmp file first, then renames to avoid partial writes.
        Raises ConstraintViolationError if config is invalid.
        """
        validate_config(config)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(config, indent=2))
        tmp_path.replace(self._path)

    def backup(self) -> str:
        """Copy current config to .bak and return the backup path."""
        bak_path = self._path.with_suffix(".bak")
        shutil.copy2(self._path, bak_path)
        return str(bak_path)

    def restore_backup(self) -> None:
        """Restore config from the most recent .bak file."""
        bak_path = self._path.with_suffix(".bak")
        if not bak_path.exists():
            raise FileNotFoundError(f"No backup found at {bak_path}")
        self._path.write_bytes(bak_path.read_bytes())

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def get_grid_shape(self) -> tuple[int, int]:
        """Return (grid_rows, grid_cols) without full validation."""
        raw = json.loads(self._path.read_text())
        return (raw["grid_rows"], raw["grid_cols"])

    def get_channels_by_role(self, role: str) -> list[tuple[int, int]]:
        """
        Return channel coordinates for the given role.

        role: "UP" | "DOWN" | "STIM" | "OFF"
        Returns a list of (row, col) tuples.
        """
        raw = json.loads(self._path.read_text())
        cols = raw.get("grid_cols", 4)
        role_map = {
            "UP":   "up_channels",
            "DOWN": "down_channels",
            "STIM": "stim_channels",
            "OFF":  "disabled_channels",
        }
        key = role_map.get(role.upper())
        if key is None:
            raise ValueError(f"Unknown role {role!r}. Valid: UP, DOWN, STIM, OFF")
        ids: list[int] = raw.get(key, [])
        return [_id_to_rc(ch_id, cols) for ch_id in ids]

    def are_adjacent(self, ch_a: tuple[int, int], ch_b: tuple[int, int]) -> bool:
        """Return True if Chebyshev distance between ch_a and ch_b is exactly 1."""
        return _chebyshev(ch_a, ch_b) == 1
