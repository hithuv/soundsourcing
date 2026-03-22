# Benchmark Walkthrough (Step by Step)

This walkthrough runs the full separation benchmark and saves all outputs (metrics, plots, comparisons, audio examples).

## Prerequisites

From project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio pandas numpy librosa scikit-learn tqdm soundfile matplotlib
```

Required data layout:

- `UrbanSound8K/` present in repository root.
- `UrbanSound8K_separation/` can be generated in step 1.
- `output/label_encoder.npy` needed for class mapping (from `train_sound_model.py` run).

## Step 1: Generate separation dataset (if missing)

```bash
python 1b_create_separation_dataset.py
```

Expected outputs:

- `UrbanSound8K_separation/audio/*.wav`
- `UrbanSound8K_separation/metadata.csv`

## Step 2: Train baseline and all upgraded models

Run the full suite:

```bash
python experiments/run_benchmark.py --epochs 30 --batch-size 4 --lr 0.001
```

This launches, in order:

1. `experiments/train/train_baseline_unet.py`
2. `experiments/train/train_resattn_unet.py`
3. `experiments/train/train_tfctdf_unet.py`
4. `experiments/train/train_conv_tasnet.py`
5. `experiments/train/train_demucs_lite.py`

Each model creates a run folder:

- `results/<timestamp>_<model>_balanced/`
  - `checkpoints/best.pth`, `checkpoints/last.pth`
  - `metrics/train_history.csv`
  - `metrics/train_summary.json`

## Step 3: Evaluate all trained runs

If you ran step 2 via `run_benchmark.py`, evaluation is already executed automatically.

Manual evaluation command:

```bash
python experiments/eval/evaluate_models.py --run-dirs results/<runA> results/<runB> results/<runC> results/<runD> results/<runE>
```

Per-run outputs:

- `results/<run_id>/metrics/per_sample.csv`
- `results/<run_id>/metrics/summary.json`
- `results/<run_id>/audio_examples/*.wav`

Aggregate outputs:

- `results/comparison/<timestamp>/leaderboard.csv`
- `results/comparison/<timestamp>/leaderboard.json`

## Step 4: Generate training and comparison plots

If you ran step 2 via `run_benchmark.py`, plotting is already executed automatically.

Manual plotting command:

```bash
python experiments/plots/generate_plots.py --run-dirs results/<runA> results/<runB> results/<runC> results/<runD> results/<runE> --comparison-dir results/comparison/<timestamp>
```

Generated artifacts:

- Per run:
  - `results/<run_id>/plots/training_curve.png`
  - `results/<run_id>/plots/lr_curve.png`
- Comparison:
  - `results/comparison/<timestamp>/model_comparison.png`
  - `results/comparison/<timestamp>/model_radar.png`
  - `results/comparison/<timestamp>/benchmark_summary.md`

## Step 5: Read model explanations and compare quality

- Architecture notes: `experiments/reports/model_explanations.md`
- Final benchmark summary: `results/comparison/<timestamp>/benchmark_summary.md`
- Listening checks: compare files in each run's `audio_examples/`.

## Optional fast sanity run

```bash
python experiments/run_benchmark.py --epochs 5 --batch-size 2 --max-eval-samples 40
```

Use this before long balanced runs to verify pipeline correctness on your machine.

