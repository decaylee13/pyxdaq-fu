# Python Files Summary and Visualization Opportunities

This report reviews the Python files in the project root that implement the neural decoding + Pong workflow. I excluded the two `conanfile.py` files under `Intan-RHX-xdaq-v1.3.0/libxdaq-dist/` because they are packaging/build config, not part of your decoding logic.

## 1) `spike_detector.py`
- **Purpose:** Converts raw amplifier samples into detected spike events using ADC-to-uV conversion, high-pass filtering, threshold crossing, and per-channel refractory lockout.
- **Inputs:** `raw_samples` array `(num_samples, num_channels)` plus chunk timestamp.
- **Outputs:** List of `SpikeEvent(channel, sample_index, time_s)`.
- **Why it matters:** This is the earliest signal-processing stage and determines detection quality for everything downstream.

**Great graph options**
- Raw vs high-pass filtered voltage trace for one channel, with threshold line.
- Spike raster plot (channel vs time) from detected events.
- Sensitivity plot: detected spike count vs threshold setting.

## 2) `spike_counter.py`
- **Purpose:** Bins spike events into fixed windows (default 10 ms) for streaming control.
- **Inputs:** Spike events (`channel`, `time_s`).
- **Outputs:** Window records with `t_start`, `t_end`, and per-channel counts.
- **Why it matters:** Establishes timing resolution and control-loop cadence.

**Great graph options**
- Per-window total spikes over time.
- Heatmap: channel (y) vs time window (x), color = spike count.

## 3) `region_mapper.py`
- **Purpose:** Aggregates channel counts into named brain regions (`up`, `down`) using sum or normalized mean.
- **Inputs:** Window count dictionary from `SpikeCounter`.
- **Outputs:** Region activity per window.
- **Why it matters:** Bridges channel-level data to behavior-level features.

**Great graph options**
- `up` vs `down` activity time-series overlay.
- Grouped bars comparing region activity for selected windows.

## 4) `action_decoder.py`
- **Purpose:** Converts region activity into discrete actions (`UP`, `DOWN`, `HOLD`) using dominance (`up - down`) and optional smoothing.
- **Inputs:** Region activity window.
- **Outputs:** Action plus dominance score per window.
- **Why it matters:** This is the decision policy mapping neural signals to control output.

**Great graph options**
- Dominance time-series with ±threshold lines.
- Action timeline (categorical strip) aligned with dominance.
- Smoothing comparison: raw dominance vs smoothed dominance.

## 5) `live_decoder.py`
- **Purpose:** End-to-end live RHX pipeline runner: receives TCP spikes, counts windows, maps regions, decodes actions, prints stream.
- **Inputs:** RHX TCP spike packets + CLI config (channels, threshold, smoothing, etc.).
- **Outputs:** Live terminal stream of action/dominance/up/down and total spikes.
- **Why it matters:** Real-time integration script proving the whole stack works online.

**Great graph options**
- Timeline of `up`, `down`, dominance, and decoded action during a live run.
- Action frequency chart (`UP`/`DOWN`/`HOLD`) per run.
- Spike throughput (spikes/sec) over time.

## 6) `latency_bench.py`
- **Purpose:** Measures real-time performance of the live pipeline and can export per-window metrics to CSV.
- **Inputs:** Live RHX stream + benchmark CLI settings.
- **Outputs:** Stats + optional CSV with `window_age_ms`, `ingestion_latency_ms`, `decode_time_us`, `interval_ms`, actions, region values.
- **Why it matters:** Best evidence for closed-loop feasibility and system timing quality.

**Great graph options (highest value for report)**
- Histogram or KDE of `window_age_ms` (end-to-end latency).
- CDF of latency (easy way to show p50/p95/p99).
- Boxplot comparison across runs (`latency.csv`, `latency2.csv`, `latency3.csv`).
- Time-series of `decode_time_us` to show compute overhead stability.

## 7) `rhx_tcp_spike_client.py`
- **Purpose:** Minimal RHX connectivity/test utility for one channel; prints incoming spike packets.
- **Inputs:** RHX host/ports/channel.
- **Outputs:** Raw spike packet fields in terminal.
- **Why it matters:** Useful for connection sanity checks before running full pipeline.

**Great graph options**
- Minimal standalone graph value.
- If desired: simple spikes-per-second trace during connection tests.

## 8) `pong_env.py`
- **Purpose:** Pure Pong physics state machine (no rendering). Handles paddle motion, collisions, scoring, and rally logic.
- **Inputs:** Action per frame (`UP`/`DOWN`/`HOLD`).
- **Outputs:** Updated `PongState`.
- **Why it matters:** Controlled simulation environment for evaluating neural control quality.

**Great graph options**
- Score progression across time.
- Rally length histogram.
- Ball trajectory sample plots (x-y paths for selected rallies).

## 9) `pong_game.py`
- **Purpose:** Playable pygame application with keyboard mode and RHX neural-control mode via background decoder thread.
- **Inputs:** Keyboard or neural action stream.
- **Outputs:** Real-time game visualization and HUD values.
- **Why it matters:** Human-visible demonstration of closed-loop control.

**Great graph options**
- Action distribution pie/bar chart per session.
- Dominance vs game outcomes (win/loss or score differential).
- Rally length comparison: keyboard mode vs RHX mode.

## 10) `test_spike_pipeline.py`
- **Purpose:** Synthetic signal demo/test for `SpikeDetector` + `SpikeCounter`, with injected spikes and validation checks.
- **Inputs:** Simulated noisy raw data with known spike injections.
- **Outputs:** Assertion-based validation and printed event/window previews.
- **Why it matters:** Demonstrates correctness of early pipeline stages.

**Great graph options**
- Injected spike times vs detected spike times (per channel).
- Confusion-style summary: expected vs detected spikes per channel/window.

## 11) `test_decoder_pipeline.py`
- **Purpose:** Synthetic tests for `RegionMapper` + `ActionDecoder` across UP/DOWN/HOLD, smoothing, and normalized mode cases.
- **Inputs:** Hand-crafted window counts.
- **Outputs:** Printed case results and asserts.
- **Why it matters:** Validates deterministic behavior of decoding policy.

**Great graph options**
- Dominance/action plots for each synthetic case.
- Small panel figure showing effect of smoothing and normalization.

---

## Best Files for Graph-Heavy Figures (Recommended)
1. **`latency_bench.py`** (and `latency*.csv`) for strongest quantitative figures.
2. **`spike_detector.py`** for signal-processing visuals (trace + threshold + spikes).
3. **`action_decoder.py` + `region_mapper.py`** for interpretable neural-to-action plots.
4. **`pong_game.py` / `pong_env.py`** for behavioral outcome visuals (scores, rallies, control quality).

## Existing Data You Can Already Plot
You currently have:
- `latency.csv` (~5,949 rows)
- `latency2.csv` (~5,990 rows)
- `latency3.csv` (~5,942 rows)

These are immediately usable for publication-style latency figures (distribution, CDF, run-to-run comparison).
