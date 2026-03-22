"""Generate publication-style figures and tables from separation benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paper-friendly defaults (IEEE/NeurIPS-style: serif, readable in print)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

MODEL_ORDER = [
    "baseline_unet",
    "resattn_unet",
    "tfctdf_unet",
    "conv_tasnet",
    "demucs_lite",
]

DISPLAY_NAMES = {
    "baseline_unet": "Baseline U-Net",
    "resattn_unet": "ResAttn U-Net",
    "tfctdf_unet": "TF–CTDF U-Net",
    "conv_tasnet": "Conv-TasNet (lite)",
    "demucs_lite": "Demucs (lite)",
}


def discover_newest_runs(results_root: Path) -> dict[str, Path]:
    """Map model name -> run directory (newest by train_summary.json mtime)."""
    pattern = re.compile(r"^\d{8}_\d{6}_(.+)_balanced$")
    best: dict[str, tuple[float, Path]] = {}
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        model = m.group(1)
        summary = child / "metrics" / "train_summary.json"
        if not summary.is_file():
            continue
        mt = summary.stat().st_mtime
        prev = best.get(model)
        if prev is None or mt > prev[0]:
            best[model] = (mt, child)
    return {k: v[1] for k, v in best.items()}


def run_eval(results_root: Path) -> Path:
    """Run SI-SDR comparison; return path to JSON."""
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    cmd = [
        sys.executable,
        str(Path("experiments/eval/compare_separation_models.py")),
        "--results-root",
        str(results_root),
    ]
    subprocess.run(cmd, check=True, cwd=Path.cwd(), env=env)
    out = results_root / "separation_model_comparison.json"
    if not out.is_file():
        raise FileNotFoundError(out)
    return out


def load_comparison_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_table_rows(
    runs: dict[str, Path],
    comparison: dict,
) -> pd.DataFrame:
    by_model = {r["model"]: r for r in comparison["ranking"]}
    rows = []
    for name in MODEL_ORDER:
        if name not in runs or name not in by_model:
            continue
        run_dir = runs[name]
        summ_path = run_dir / "metrics" / "train_summary.json"
        summ = json.loads(summ_path.read_text(encoding="utf-8"))
        ev = by_model[name]
        rows.append(
            {
                "model_key": name,
                "model": DISPLAY_NAMES.get(name, name),
                "best_val_loss_train": summ["best_val_loss"],
                "best_epoch": summ["best_epoch"],
                "epochs_ran": summ["epochs_ran"],
                "elapsed_hours": summ.get("elapsed_sec", 0) / 3600.0,
                "mean_si_sdr_db": ev["mean_si_sdr_db"],
                "mean_waveform_l1_eval": ev["mean_waveform_l1"],
                "si_sdr_pairs": ev.get("si_sdr_pairs", ""),
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(rows)


def to_latex_table(df: pd.DataFrame) -> str:
    """Minimal LaTeX table (requires \\usepackage{booktabs} in the preamble)."""
    lines = [
        r"% \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Separation models on the UrbanSound8K mixture validation split. "
        r"Training loss is the objective in \texttt{train\_utils} (spectrogram vs.\ waveform). "
        r"SI-SDR is mean over active class channels (RMS $> 10^{-3}$).}",
        r"\label{tab:separation-benchmark}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Best val.\ loss & SI-SDR (dB) & WF L1 (eval) & Best ep. \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']} & {r['best_val_loss_train']:.4f} & {r['mean_si_sdr_db']:.3f} & "
            f"{r['mean_waveform_l1_eval']:.4f} & {int(r['best_epoch'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def plot_training_curves(runs: dict[str, Path], out_dir: Path) -> None:
    n = len([m for m in MODEL_ORDER if m in runs])
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.4 * rows), constrained_layout=True)
    ax_flat = np.atleast_1d(axes).ravel()
    idx = 0
    for name in MODEL_ORDER:
        if name not in runs:
            continue
        ax = ax_flat[idx]
        idx += 1
        csv_path = runs[name] / "metrics" / "train_history.csv"
        df = pd.read_csv(csv_path)
        ep = df["epoch"].values
        ax.plot(ep, df["train_loss"], label="Train", color="#1f77b4", linewidth=1.2)
        ax.plot(ep, df["val_loss"], label="Val", color="#d62728", linewidth=1.2)
        ax.set_title(DISPLAY_NAMES.get(name, name))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", framealpha=0.9)
    for j in range(idx, len(ax_flat)):
        ax_flat[j].axis("off")
    fig.suptitle("Training and validation loss by model", fontsize=11, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_training_curves.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_si_sdr_bars(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    order = df.sort_values("mean_si_sdr_db", ascending=True)
    y = np.arange(len(order))
    vals = order["mean_si_sdr_db"].values
    labels = order["model"].values
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(order)))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean SI-SDR (dB)")
    ax.set_title("Waveform SI-SDR on validation (active sources)")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_si_sdr_validation.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper tables and figures.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper"))
    parser.add_argument(
        "--re-eval",
        action="store_true",
        help="Re-run SI-SDR evaluation (slow). Default: use results/separation_model_comparison.json if present.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_newest_runs(results_root)
    missing = [m for m in MODEL_ORDER if m not in runs]
    if missing:
        print("Warning: missing training runs for:", ", ".join(missing))

    comp_path = results_root / "separation_model_comparison.json"
    if args.re_eval or not comp_path.is_file():
        print("Running SI-SDR evaluation (may take several minutes)...")
        comp_path = run_eval(results_root)
    else:
        print(f"Using existing {comp_path} (pass --re-eval to recompute)")
    comparison = load_comparison_json(comp_path)

    df = build_table_rows(runs, comparison)
    csv_path = out_dir / "table_benchmark.csv"
    df.to_csv(csv_path, index=False)
    tex_path = out_dir / "table_benchmark.tex"
    tex_path.write_text(to_latex_table(df), encoding="utf-8")
    json_path = out_dir / "benchmark_summary.json"
    json_path.write_text(
        json.dumps({"comparison": comparison, "training_runs": {k: str(v) for k, v in runs.items()}}, indent=2),
        encoding="utf-8",
    )

    plot_training_curves(runs, out_dir)
    plot_si_sdr_bars(df, out_dir)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")
    print(f"Wrote: {json_path}")
    print(f"Figures in: {out_dir} (PDF + PNG)")


if __name__ == "__main__":
    main()
