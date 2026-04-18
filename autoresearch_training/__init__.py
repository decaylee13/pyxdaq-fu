"""
autoresearch_training

Searches stimulation parameter space to find feedback regimes that promote
genuine learning in a cultured MEA (multi-electrode array), operating on a
multi-session timescale distinct from single-session configuration optimisation.

This library is a sibling to autoresearch/, which finds the best static electrode
configuration (channel roles) for a given culture.  autoresearch_training/ is run
after autoresearch/ has locked in the optimal electrode layout: it then searches
over reward and penalty pulse characteristics (amplitude, duration, pulse count,
inter-stimulus interval) to find the stimulation regime that produces the steepest
drift-corrected improvement in mean rally length across repeated game sessions.

The central methodological challenge addressed here is isolating genuine learning
(performance improvement driven by stimulation feedback) from biological drift
(spontaneous changes in network state unrelated to the game).  This requires:

  - Running multiple sessions per regime to estimate a learning slope
  - Periodically running no-stim control sessions to characterise the drift baseline
  - Applying subtract_drift() to remove the drift component before any keep/discard decision

Typical usage:
    from autoresearch_training import TrainingResearcher

    researcher = TrainingResearcher(
        project_root=".",
        session_seconds=180,
        sessions_per_regime=6,
        min_sessions_before_decision=6,
        control_interval=5,
    )
    researcher.setup()
    researcher.run()
    print(researcher.summarize())
"""

from autoresearch_training.researcher import TrainingResearcher
from autoresearch_training.stimulation import StimRegime
from autoresearch_training.learning import LearningCurve

__all__ = ["TrainingResearcher", "StimRegime", "LearningCurve"]
