"""
autoresearch — autonomous experiment loop for pyxdaq-fu BCI Pong.

This library implements a self-contained optimization loop that iteratively
improves electrode configurations and decoder parameters for a neural-controlled
Pong game.  It reads and writes ``channel_config.json`` and a small set of
ActionDecoder parameters, launches timed game sessions against live RHX
hardware (or a mock controller for offline testing), measures performance via
mean rally length and hit rate, and uses a git-backed keep/revert strategy
to accumulate only improving experiments.  All config writes are validated
against hard biological constraints (minimum region sizes, STIM/REC separation)
before being committed to disk.

Quick start::

    from autoresearch import AutoResearcher

    researcher = AutoResearcher(project_root=".", session_seconds=90, n_repeats=2)
    researcher.setup()   # run baseline, validate hardware, print layout
    researcher.run()     # loop until Ctrl-C
    researcher.summarize()
"""

from autoresearch.researcher import AutoResearcher
from autoresearch.experiment_runner import run_session
from autoresearch.metrics import SessionResult

__all__ = ["AutoResearcher", "run_session", "SessionResult"]
