"""Generate per-run and cross-run plots for separation benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark plots.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--comparison-dir", default="", help="Folder containing leaderboard.csv (optional).")
    return parser.parse_args()


def plot_training_curves(run_dir: Path) -> None:
    history_path = run_dir / "metrics" / "train_history.csv"
    if not history_path.exists():
        return
    df = pd.read_csv(history_path)
    if df.empty:
        return
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve - {run_dir.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["epoch"], df["lr"], label="learning_rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title(f"LR Schedule - {run_dir.name}")
    plt.tight_layout()
    plt.savefig(out_dir / "lr_curve.png", dpi=160)
    plt.close()


def plot_metric_bars(comparison_dir: Path) -> None:
    leaderboard_path = comparison_dir / "leaderboard.csv"
    if not leaderboard_path.exists():
        return
    leaderboard = pd.read_csv(leaderboard_path)
    if leaderboard.empty:
        return
    leaderboard = leaderboard.sort_values("mean_si_sdr", ascending=False)

    plt.figure(figsize=(9, 5))
    x = range(len(leaderboard))
    plt.bar(x, leaderboard["mean_si_sdr"], label="SI-SDR")
    plt.bar(x, leaderboard["mean_sdr"], alpha=0.45, label="SDR")
    plt.xticks(x, leaderboard["model"], rotation=25, ha="right")
    plt.ylabel("dB")
    plt.title("Model Comparison (Higher is Better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "model_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 7))
    for _, row in leaderboard.iterrows():
        values = [row["mean_si_sdr"], row["mean_sdr"], row["mean_sir"], row["mean_si_sdr"]]
        plt.polar([0, 2.09, 4.19, 0], values, marker="o", label=row["model"])
    plt.title("Radar Comparison (SI-SDR / SDR / SIR)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    plt.savefig(comparison_dir / "model_radar.png", dpi=180)
    plt.close()


def write_summary_markdown(comparison_dir: Path) -> None:
    leaderboard_path = comparison_dir / "leaderboard.csv"
    if not leaderboard_path.exists():
        return
    leaderboard = pd.read_csv(leaderboard_path).sort_values("mean_si_sdr", ascending=False)
    markdown = ["# Separation Benchmark Results", "", "## Leaderboard", ""]
    markdown.append("| Rank | Model | Mean SI-SDR | Mean SDR | Mean SIR |")
    markdown.append("|---:|---|---:|---:|---:|")
    for rank, row in enumerate(leaderboard.itertuples(index=False), start=1):
        markdown.append(
            f"| {rank} | {row.model} | {row.mean_si_sdr:.3f} | {row.mean_sdr:.3f} | {row.mean_sir:.3f} |"
        )
    (comparison_dir / "benchmark_summary.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in args.run_dirs]
    for run_dir in run_dirs:
        plot_training_curves(run_dir)

    if args.comparison_dir:
        comparison_dir = Path(args.comparison_dir)
    else:
        inferred = sorted(Path("results/comparison").glob("*"), reverse=True)
        comparison_dir = inferred[0] if inferred else Path("")
    if comparison_dir and comparison_dir.exists():
        plot_metric_bars(comparison_dir)
        write_summary_markdown(comparison_dir)
        print(json.dumps({"comparison_dir": str(comparison_dir)}, indent=2))


if __name__ == "__main__":
    main()

