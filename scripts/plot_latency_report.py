#!/usr/bin/env python3
"""
Generate report-ready figures and summary stats from latency benchmark CSV files.

Default behavior scans for latency*.csv in the current directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("window_age_ms", "ingestion_latency_ms", "decode_time_us")
ACTION_CLASSES = ("UP", "DOWN", "HOLD")


@dataclass
class RunData:
    label: str
    path: Path
    window: np.ndarray
    action: np.ndarray
    window_age_ms: np.ndarray
    ingestion_latency_ms: np.ndarray
    decode_time_us: np.ndarray

    def metric(self, name: str) -> np.ndarray:
        if name == "window_age_ms":
            return self.window_age_ms
        if name == "ingestion_latency_ms":
            return self.ingestion_latency_ms
        if name == "decode_time_us":
            return self.decode_time_us
        raise KeyError(f"Unknown metric: {name}")


def _parse_float(raw: str | None) -> float:
    if raw is None:
        return np.nan
    txt = raw.strip()
    if not txt:
        return np.nan
    try:
        return float(txt)
    except ValueError:
        return np.nan


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    txt = raw.strip()
    if not txt:
        return default
    try:
        return int(float(txt))
    except ValueError:
        return default


def load_run(path: Path) -> RunData:
    windows: list[int] = []
    actions: list[str] = []
    window_age: list[float] = []
    ingestion: list[float] = []
    decode: list[float] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            windows.append(_parse_int(row.get("window"), idx))
            actions.append((row.get("action") or "HOLD").strip().upper())
            window_age.append(_parse_float(row.get("window_age_ms")))
            ingestion.append(_parse_float(row.get("ingestion_latency_ms")))
            decode.append(_parse_float(row.get("decode_time_us")))

    return RunData(
        label=path.stem,
        path=path,
        window=np.asarray(windows, dtype=np.int64),
        action=np.asarray(actions, dtype=object),
        window_age_ms=np.asarray(window_age, dtype=np.float64),
        ingestion_latency_ms=np.asarray(ingestion, dtype=np.float64),
        decode_time_us=np.asarray(decode, dtype=np.float64),
    )


def finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def describe(a: np.ndarray) -> dict[str, float]:
    x = finite(a)
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }
    return {
        "n": float(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def write_summary(runs: list[RunData], out_csv: Path) -> None:
    rows: list[dict[str, str | float | int]] = []
    for run in runs:
        for metric in METRICS:
            s = describe(run.metric(metric))
            rows.append(
                {
                    "run": run.label,
                    "metric": metric,
                    "n": int(s["n"]),
                    "mean": s["mean"],
                    "std": s["std"],
                    "p50": s["p50"],
                    "p95": s["p95"],
                    "p99": s["p99"],
                }
            )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "metric", "n", "mean", "std", "p50", "p95", "p99"],
        )
        writer.writeheader()
        writer.writerows(rows)


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    step = max(1, x.size // max_points)
    return x[::step], y[::step]


def plot_window_age_hist(runs: list[RunData], out_png: Path) -> None:
    all_vals = [finite(r.window_age_ms) for r in runs]
    all_vals = [x for x in all_vals if x.size > 0]
    if not all_vals:
        return

    merged = np.concatenate(all_vals)
    low = float(np.percentile(merged, 1))
    high = float(np.percentile(merged, 99))
    if low == high:
        low -= 1.0
        high += 1.0
    bins = np.linspace(low, high, 60)

    plt.figure(figsize=(10, 5.5))
    for run in runs:
        x = finite(run.window_age_ms)
        if x.size == 0:
            continue
        plt.hist(x, bins=bins, density=True, alpha=0.35, label=run.label)
    plt.xlabel("window_age_ms")
    plt.ylabel("Density")
    plt.title("End-to-End Latency Distribution (window_age_ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_window_age_cdf(runs: list[RunData], out_png: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    for run in runs:
        x = np.sort(finite(run.window_age_ms))
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
        plt.plot(x, y, linewidth=1.7, label=run.label)

    plt.xlabel("window_age_ms")
    plt.ylabel("CDF")
    plt.title("End-to-End Latency CDF")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_window_age_boxplot(runs: list[RunData], out_png: Path) -> None:
    labels: list[str] = []
    values: list[np.ndarray] = []
    for run in runs:
        x = finite(run.window_age_ms)
        if x.size == 0:
            continue
        labels.append(run.label)
        values.append(x)
    if not values:
        return

    plt.figure(figsize=(9.5, 5.5))
    plt.boxplot(values, tick_labels=labels, showfliers=False)
    plt.ylabel("window_age_ms")
    plt.title("Latency Comparison Across Runs")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_timeseries(runs: list[RunData], out_png: Path, max_points: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_age, ax_decode = axes

    for run in runs:
        age = finite(run.window_age_ms)
        dec = finite(run.decode_time_us)
        if age.size > 0:
            x_age = np.arange(1, age.size + 1, dtype=np.int64)
            x_age, age_ds = downsample_xy(x_age, age, max_points=max_points)
            ax_age.plot(x_age, age_ds, linewidth=0.8, alpha=0.9, label=run.label)
        if dec.size > 0:
            x_dec = np.arange(1, dec.size + 1, dtype=np.int64)
            x_dec, dec_ds = downsample_xy(x_dec, dec, max_points=max_points)
            ax_decode.plot(x_dec, dec_ds, linewidth=0.8, alpha=0.9, label=run.label)

    ax_age.set_ylabel("window_age_ms")
    ax_age.set_title("Per-Window Latency Over Time")
    ax_age.grid(alpha=0.2)
    ax_age.legend()

    ax_decode.set_ylabel("decode_time_us")
    ax_decode.set_xlabel("Window index")
    ax_decode.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_action_distribution(runs: list[RunData], out_png: Path) -> None:
    if not runs:
        return

    labels = [r.label for r in runs]
    y = np.arange(len(runs))
    fractions_by_class: dict[str, list[float]] = {k: [] for k in ACTION_CLASSES}
    for run in runs:
        n = max(1, run.action.size)
        for action in ACTION_CLASSES:
            fractions_by_class[action].append(float(np.sum(run.action == action)) * 100.0 / n)

    colors = {"UP": "#2ca02c", "DOWN": "#d62728", "HOLD": "#7f7f7f"}
    plt.figure(figsize=(10, 5.5))
    left = np.zeros(len(runs), dtype=np.float64)
    for action in ACTION_CLASSES:
        vals = np.asarray(fractions_by_class[action], dtype=np.float64)
        plt.barh(y, vals, left=left, color=colors[action], alpha=0.85, label=action)
        left += vals

    plt.yticks(y, labels)
    plt.xlim(0, 100)
    plt.xlabel("Percent of windows")
    plt.title("Decoded Action Distribution by Run")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def detect_default_inputs() -> list[Path]:
    return sorted(Path(".").glob("latency*.csv"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate latency benchmark figures from CSV files.")
    p.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Input CSV files. Defaults to auto-detected latency*.csv in current directory.",
    )
    p.add_argument("--outdir", default="latency_figures", help="Output directory for figures and summary.")
    p.add_argument("--max-points", type=int, default=3000, help="Max points per line in time-series plots.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.inputs:
        input_paths = [Path(x) for x in args.inputs]
    else:
        input_paths = detect_default_inputs()

    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {', '.join(missing)}")
    if not input_paths:
        raise FileNotFoundError("No input CSVs found. Provide --inputs or place latency*.csv in this directory.")

    runs = [load_run(p) for p in input_paths]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_window_age_hist(runs, outdir / "window_age_hist.png")
    plot_window_age_cdf(runs, outdir / "window_age_cdf.png")
    plot_window_age_boxplot(runs, outdir / "window_age_boxplot.png")
    plot_timeseries(runs, outdir / "latency_timeseries.png", max_points=args.max_points)
    plot_action_distribution(runs, outdir / "action_distribution.png")
    write_summary(runs, outdir / "latency_summary.csv")

    print("Generated files:")
    print(f"  {outdir / 'window_age_hist.png'}")
    print(f"  {outdir / 'window_age_cdf.png'}")
    print(f"  {outdir / 'window_age_boxplot.png'}")
    print(f"  {outdir / 'latency_timeseries.png'}")
    print(f"  {outdir / 'action_distribution.png'}")
    print(f"  {outdir / 'latency_summary.csv'}")


if __name__ == "__main__":
    main()
