"""Run the full benchmark suite end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TRAIN_SCRIPTS = [
    "experiments/train/train_baseline_unet.py",
    "experiments/train/train_resattn_unet.py",
    "experiments/train/train_tfctdf_unet.py",
    "experiments/train/train_conv_tasnet.py",
    "experiments/train/train_demucs_lite.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute full separation benchmark.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


def call_python(script: str, args: list[str]) -> None:
    cmd = [sys.executable, script] + args
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def newest_model_run_dirs(output_root: str) -> list[str]:
    root = Path(output_root)
    run_dirs = []
    for prefix in ["baseline_unet", "resattn_unet", "tfctdf_unet", "conv_tasnet", "demucs_lite"]:
        matches = sorted(root.glob(f"*_{prefix}_balanced"))
        if not matches:
            raise FileNotFoundError(f"No run dir found for {prefix} in {root}")
        run_dirs.append(str(matches[-1]))
    return run_dirs


def main() -> None:
    args = parse_args()
    common_args = [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--output-root",
        args.output_root,
    ]
    for script in TRAIN_SCRIPTS:
        call_python(script, common_args)

    run_dirs = newest_model_run_dirs(args.output_root)
    eval_args = ["--run-dirs", *run_dirs, "--max-samples", str(args.max_eval_samples)]
    call_python("experiments/eval/evaluate_models.py", eval_args)

    comparison_dirs = sorted(Path("results/comparison").glob("*"))
    comparison_dir = str(comparison_dirs[-1]) if comparison_dirs else ""
    plot_args = ["--run-dirs", *run_dirs]
    if comparison_dir:
        plot_args.extend(["--comparison-dir", comparison_dir])
    call_python("experiments/plots/generate_plots.py", plot_args)


if __name__ == "__main__":
    main()

