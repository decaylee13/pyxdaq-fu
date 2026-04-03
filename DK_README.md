# Command to running the software: 
- open ./Intan-RHX-xdaq-v1.3.0/build/build/Release/XDAQ-RHX.app

# RHX TCP Test program: 
- python rhx_tcp_spike_client.py --channel A-000 --duration-s 60 --cleanup

# Pong Game 
- Live neural input: python pong_game.py --mode rhx
- Play against computer: python pong_game.py --mode keyboard

# Running the PONG game
- python pong_game.py --mode rhx --threshold 0.5 --smooth-windows 1

# Selecting the channels 
- python channel_selector.py 

# Latency report figures
- source .venv/bin/activate
- python scripts/plot_latency_report.py

# Latency 1 -> 4 interpretation (for report)
Across the four latency runs (latency, latency2, latency3, latency4), the end-to-end window age improved overall with one temporary regression: mean `window_age_ms` moved from `356.2` (run 1) to `37.8` (run 2), rose to `284.0` (run 3), then dropped to `22.0` in run 4, while p99 improved from `670.7` to `48.3` by run 4. This leftward shift and tightening in the CDF/boxplot indicates that the latest pipeline is both faster and more stable, with ~`93.8%` lower mean latency and ~`92.8%` lower p99 latency versus run 1.

The code fixes aligned with this change were timing and transport fixes in the live path: we anchored RHX time to wall time on first packet and aligned the first window start to that RHX timestamp (`counter._next_window_start = rhx_time_s`) so window boundaries are consistent ([latency_bench.py](/Users/dklee/Documents/pyxdaq-fu/latency_bench.py#L246), [latency_bench.py](/Users/dklee/Documents/pyxdaq-fu/latency_bench.py#L249)); we switched window draining to use wall-clock-derived RHX time (`current_time_s = time.perf_counter() - wall_offset`) so windows keep draining during sparse-spike periods instead of waiting for the next spike burst ([latency_bench.py](/Users/dklee/Documents/pyxdaq-fu/latency_bench.py#L263), [latency_bench.py](/Users/dklee/Documents/pyxdaq-fu/latency_bench.py#L264)); and we hardened TCP behavior by connecting the spike socket before status polling to avoid pending/connected deadlock and by adding packet re-sync on bad magic for stream recovery ([latency_bench.py](/Users/dklee/Documents/pyxdaq-fu/latency_bench.py#L186), [latency_bench.py](/Users/dklee/Documents/pyxdaq-fu/latency_bench.py#L239)). Decode compute time stayed low (single-digit to low tens of microseconds), which supports that the primary gain came from reducing staleness and jitter rather than speeding up mapping/decoding math.
