#!/usr/bin/env python3
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

# Support both:
# - python -m pyxdaq.test
# - python pyxdaq/test.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pyxdaq.datablock import Samples
    from pyxdaq.simulated_xdaq import SimulatedXDAQ
    from pyxdaq.writer import OpenEphysWriter
else:
    from .datablock import Samples
    from .simulated_xdaq import SimulatedXDAQ
    from .writer import OpenEphysWriter


@dataclass
class StreamStats:
    callback_count: int = 0
    sample_count: int = 0
    first_sample_index: Optional[int] = None
    last_sample_index: Optional[int] = None
    error_count: int = 0
    last_error: Optional[str] = None


def main():
    parser = argparse.ArgumentParser(description="Simulated XDAQ step-4 smoke test")
    parser.add_argument("--rhs", action="store_true", help="Use RHS sample layout")
    parser.add_argument("--streams", type=int, default=2, help="Number of simulated data streams")
    parser.add_argument("--sample-rate", type=int, default=30000, help="Sampling rate in Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="Run duration in seconds")
    parser.add_argument("--chunk-samples", type=int, default=128, help="Samples per callback chunk")
    parser.add_argument(
        "--record-root",
        type=Path,
        default=None,
        help="If set, write Open Ephys output under this directory",
    )
    args = parser.parse_args()

    xdaq = SimulatedXDAQ(rhs=args.rhs, num_streams=args.streams, sample_rate=args.sample_rate)
    stats = StreamStats()

    writer_cm = (
        OpenEphysWriter(xdaq, root_path=str(args.record_root))
        if args.record_root is not None
        else None
    )

    def on_error(error: str):
        stats.error_count += 1
        stats.last_error = error
        print(f"[stream-error] {error}")

    def on_samples_received(samples: Samples):
        stats.callback_count += 1
        stats.sample_count += samples.n
        first = int(samples.sample_index[0])
        last = int(samples.sample_index[-1])
        if stats.first_sample_index is None:
            stats.first_sample_index = first
        stats.last_sample_index = last
        if writer_cm is not None:
            writer_cm.write_data(samples)

    t0 = time.time()
    if writer_cm is not None:
        writer_cm.start_recording()
    try:
        with xdaq.start_receiving_samples(
            callbacks=[on_samples_received],
            on_error=on_error,
            chunk_samples=args.chunk_samples,
        ):
            xdaq.start(continuous=True)
            while time.time() - t0 < args.duration:
                time.sleep(0.01)
            xdaq.stop(wait=True)
    finally:
        if writer_cm is not None:
            writer_cm.stop_recording()

    elapsed = max(time.time() - t0, 1e-9)
    print("\n=== Simulated XDAQ Smoke Test ===")
    print(f"rhs: {args.rhs}")
    print(f"streams: {xdaq.numDataStream}")
    print(f"sample_rate_hz: {xdaq.getSampleRate()}")
    print(f"duration_s: {elapsed:.3f}")
    print(f"callbacks: {stats.callback_count}")
    print(f"samples_received: {stats.sample_count}")
    print(f"effective_rate_hz: {stats.sample_count / elapsed:.2f}")
    print(f"sample_index_first: {stats.first_sample_index}")
    print(f"sample_index_last: {stats.last_sample_index}")
    print(f"errors: {stats.error_count}")
    if stats.last_error is not None:
        print(f"last_error: {stats.last_error}")
    if writer_cm is not None:
        print(f"record_root: {args.record_root.resolve()}")


if __name__ == "__main__":
    main()
