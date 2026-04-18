# pyxdaq-fu autoresearch_training — Research Agenda

## Objective
Find the stimulation parameter regime (reward and penalty pulse characteristics)
that produces the steepest drift-corrected improvement in mean rally length
across repeated game sessions. This measures whether the culture is genuinely
learning from the feedback signal, not just performing well at a point in time.

## Session Configuration
- session_seconds: [FILL IN — recommended starting point: 120–300 sec]
- sessions_per_regime: [FILL IN — recommended: 5–10 before deciding]
- min_sessions_before_decision: [FILL IN — recommended: same as sessions_per_regime]
- control_interval: [FILL IN — recommended: every 5 sessions]

## Culture Notes
- Date prepared: [FILL IN]
- Baseline stim parameters (from stimulator.py defaults): [FILL IN after setup()]
- Observed spontaneous firing rate (no stim): [FILL IN after setup()]
- Baseline mean_rally_length: [FILL IN after setup()]
- Estimated drift rate from initial control sessions: [FILL IN after setup()]

## Known Constraints for This Culture
- Any electrodes to avoid for STIM: [FILL IN]
- Any known sensitivity to high-amplitude pulses: [FILL IN]

## Hypotheses to Explore
- [ ] Does stronger reward amplitude accelerate learning more than longer duration?
- [ ] Is penalty stimulation necessary, or does reward alone suffice?
- [ ] Does a higher reward:penalty ratio produce faster learning?
- [ ] Do burst pulses (multiple pulses per event) outperform single pulses?
- [ ] Is there an optimal inter-stimulus interval within bursts?

## Session History & Findings
<!-- TrainingResearcher appends summaries here every 20 regimes -->
