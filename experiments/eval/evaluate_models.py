"""Unified evaluator for all benchmark separation models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.common.metrics import sdr, si_sdr, sir
from experiments.configs.defaults import BALANCED_PRESET
from experiments.data.separation_dataset import AudioConfig, SeparationDataset, build_class_map, load_split_metadata
from experiments.train.train_utils import MODEL_FACTORIES, SPEC_MODELS, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained separation models.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="One or more results/<run_id> directories.")
    parser.add_argument("--output-dir", default="", help="Optional output folder for aggregate comparison.")
    parser.add_argument("--seed", type=int, default=BALANCED_PRESET.random_seed)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate full test split.")
    return parser.parse_args()


def locate_checkpoint(run_dir: Path) -> Path:
    for candidate in [run_dir / "checkpoints" / "best.pth", run_dir / "checkpoints" / "last.pth"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {run_dir}/checkpoints")


def _mag_to_wave(magnitude: np.ndarray, phase: np.ndarray, hop_length: int, target_len: int) -> np.ndarray:
    wave = librosa.istft(magnitude * phase, hop_length=hop_length)
    if len(wave) < target_len:
        wave = np.pad(wave, (0, target_len - len(wave)))
    return wave[:target_len]


def evaluate_run(run_dir: Path, device: torch.device, max_samples: int) -> dict[str, object]:
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_name = run_info["model"]
    class_map = build_class_map()
    idx_to_class = {v: k for k, v in class_map.items()}
    audio_cfg = AudioConfig(
        sample_rate=config["sample_rate"],
        samples=config["sample_rate"] * config["duration_sec"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        num_classes=config["num_classes"],
    )
    split = load_split_metadata(
        Path(config["data_dir"]) / "metadata.csv",
        config["val_ratio"],
        config["test_ratio"],
        config["random_seed"],
    )
    test_ds = SeparationDataset(
        split["test"],
        Path(config["data_dir"]) / "audio",
        class_map,
        config["original_metadata_path"],
        audio_cfg,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = MODEL_FACTORIES[model_name](audio_cfg.num_classes).to(device)
    model.load_state_dict(torch.load(locate_checkpoint(run_dir), map_location=device))
    model.eval()

    metrics_rows = []
    saved_examples = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc=f"evaluate {model_name}")):
            if max_samples and idx >= max_samples:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            mix_wave = batch["mix_wave"][0].cpu().numpy()
            target_waves = batch["source_waves"][0].cpu().numpy()
            mix_mag = batch["mix_mag"].to(device)

            if model_name in SPEC_MODELS:
                predicted_masks = model(mix_mag)[0].cpu().numpy()
                mag = mix_mag[0, 0].cpu().numpy()
                stft_mix = librosa.stft(mix_wave, n_fft=audio_cfg.n_fft, hop_length=audio_cfg.hop_length)
                _, phase = librosa.magphase(stft_mix)
                pred_waves = np.zeros_like(target_waves)
                for class_idx in range(audio_cfg.num_classes):
                    mask = predicted_masks[class_idx]
                    h, w = mask.shape
                    pred_mag = mag[:h, :w] * mask
                    pred_waves[class_idx] = _mag_to_wave(pred_mag, phase[:h, :w], audio_cfg.hop_length, audio_cfg.samples)
            else:
                pred_waves = model(batch["mix_wave"]).cpu().numpy()[0]
                pred_waves = pred_waves[:, : audio_cfg.samples]

            file_name = batch["mix_filename"][0]
            active_classes = np.where(np.mean(np.abs(target_waves), axis=1) > 1e-4)[0]
            for class_idx in active_classes:
                ref = target_waves[class_idx]
                est = pred_waves[class_idx]
                metrics_rows.append(
                    {
                        "model": model_name,
                        "mix_filename": file_name,
                        "class_idx": int(class_idx),
                        "class_name": idx_to_class.get(int(class_idx), str(class_idx)),
                        "sdr": sdr(ref, est),
                        "si_sdr": si_sdr(ref, est),
                        "sir": sir(ref, est, mix_wave),
                    }
                )

            if saved_examples < config["save_audio_examples"]:
                out_dir = run_dir / "audio_examples"
                sf.write(out_dir / f"{file_name}_mix.wav", mix_wave, audio_cfg.sample_rate)
                for class_idx in active_classes[:2]:
                    sf.write(out_dir / f"{file_name}_target_{idx_to_class[class_idx]}.wav", target_waves[class_idx], audio_cfg.sample_rate)
                    sf.write(out_dir / f"{file_name}_pred_{idx_to_class[class_idx]}.wav", pred_waves[class_idx], audio_cfg.sample_rate)
                saved_examples += 1

    per_sample_df = pd.DataFrame(metrics_rows)
    per_sample_df.to_csv(run_dir / "metrics" / "per_sample.csv", index=False)
    summary = {
        "model": model_name,
        "num_rows": int(len(per_sample_df)),
        "mean_sdr": float(per_sample_df["sdr"].mean()) if len(per_sample_df) else float("nan"),
        "mean_si_sdr": float(per_sample_df["si_sdr"].mean()) if len(per_sample_df) else float("nan"),
        "mean_sir": float(per_sample_df["sir"].mean()) if len(per_sample_df) else float("nan"),
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()
    run_dirs = [Path(path) for path in args.run_dirs]
    summaries = [evaluate_run(run_dir, device, args.max_samples) for run_dir in run_dirs]

    aggregate_dir = Path(args.output_dir) if args.output_dir else (Path("results") / "comparison" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame(summaries).sort_values(by="mean_si_sdr", ascending=False)
    leaderboard.to_csv(aggregate_dir / "leaderboard.csv", index=False)
    (aggregate_dir / "leaderboard.json").write_text(leaderboard.to_json(orient="records", indent=2), encoding="utf-8")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()

