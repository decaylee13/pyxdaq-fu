# pyxdaq-fu Autoresearch — Research Agenda

## Objective
Maximize mean_rally_length. This measures sustained paddle control,
which requires the neurons to maintain coherent UP/DOWN firing differentiation.

A high mean_rally_length means:
- The neural culture is producing distinct, separable activity patterns
- The electrode layout is capturing those patterns cleanly
- The decoder threshold is well-matched to the culture's dynamic range

## Current Culture Notes
- Date prepared: [FILL IN]
- Known dead channels: [FILL IN, e.g. (3,2), (7,0)]
- Observed dominant firing region: [FILL IN after first session]
- Baseline mean_rally_length: [FILL IN after setup()]
- Baseline hit_rate: [FILL IN after setup()]
- Baseline peak_firing_hz: [FILL IN after setup()]

## Constraints and Known Issues
- Channels excluded from autoresearch: [FILL IN if any channels have noisy amplifiers]
- Estimated culture health: [FILL IN — excellent / good / degrading]
- Last media change: [FILL IN]

## Hypotheses to Explore
- [ ] Are UP/DOWN regions spatially separated or interleaved in this culture?
- [ ] Are current STIM channels close enough to active recording regions?
- [ ] Is the dominance threshold well-matched to actual firing rates?
- [ ] Are any channels adding noise without contributing signal?
- [ ] Does increasing smooth_windows improve rally stability?
- [ ] Is the AI paddle speed calibrated to give the neurons a fair challenge?

## Decoder Parameter Notes
- threshold (default 3.0 Hz): raise if too many spurious UP/DOWN triggers;
  lower if the paddle is not moving enough
- smooth_windows (default 1): increase to 3–5 if the action is jittery;
  keep at 1 if response latency is more important than stability

## Session History
<!-- AutoResearcher appends summaries here after every 20 experiments -->
