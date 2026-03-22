"""Compare trained separation checkpoints on a shared metric (SI-SDR on waveforms)."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.configs.defaults import BALANCED_PRESET, ExperimentConfig
from experiments.data.separation_dataset import AudioConfig, SeparationDataset, build_class_map, load_split_metadata
from experiments.train.train_utils import MODEL_FACTORIES, SPEC_MODELS, select_device


def _discover_checkpoints(results_root: Path) -> dict[str, Path]:
    """Pick the newest run directory per model name (folder name contains ``_<model>_``)."""
    best_by_model: dict[str, tuple[float, Path]] = {}
    pattern = re.compile(r"^\d{8}_\d{6}_(.+)_balanced$")
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        model_name = m.group(1)
        ckpt = child / "checkpoints" / "best.pth"
        if not ckpt.is_file():
            continue
        mtime = ckpt.stat().st_mtime
        prev = best_by_model.get(model_name)
        if prev is None or mtime > prev[0]:
            best_by_model[model_name] = (mtime, ckpt)
    return {name: path for name, (_, path) in best_by_model.items()}


def si_sdr_numpy(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR (SI-SDR) in dB for single-channel 1D signals."""
    estimate = estimate.astype(np.float64)
    reference = reference.astype(np.float64)
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    ref_pow = np.sum(reference**2) + eps
    scale = np.sum(estimate * reference) / ref_pow
    s_target = scale * reference
    e_noise = estimate - s_target
    num = np.sum(s_target**2) + eps
    den = np.sum(e_noise**2) + eps
    return float(10.0 * np.log10(num / den))


def _active_mask(reference: np.ndarray, thresh: float) -> bool:
    rms = float(np.sqrt(np.mean(reference**2) + 1e-12))
    return rms > thresh


def mag_mask_to_waveform(
    mix_wave: np.ndarray,
    mix_mag: np.ndarray,
    mask: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """ISTFT of (mix_mag * mask) with mixture phase — matches training reconstruction."""
    stft = librosa.stft(mix_wave, n_fft=n_fft, hop_length=hop_length)
    if mix_mag.ndim == 3:
        mix_mag = mix_mag[0]
    h = min(stft.shape[0], mask.shape[0], mix_mag.shape[0])
    w = min(stft.shape[1], mask.shape[1], mix_mag.shape[1])
    stft_c = stft[:h, :w]
    mm = mix_mag[:h, :w]
    m = np.clip(mask[:h, :w], 0.0, 1.0)
    phase = np.angle(stft_c)
    est = (mm * m) * np.exp(1.0j * phase)
    out = librosa.istft(est, hop_length=hop_length, length=len(mix_wave))
    return out.astype(np.float32)


def evaluate_model(
    model_name: str,
    checkpoint: Path,
    config: ExperimentConfig,
    val_loader: DataLoader,
    device: torch.device,
    ref_rms_thresh: float,
) -> dict[str, float]:
    num_classes = config.num_classes
    model = MODEL_FACTORIES[model_name](num_classes).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    sum_sdr = 0.0
    count_sdr = 0
    sum_l1 = 0.0
    count_l1 = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"eval {model_name}"):
            mix_wave = batch["mix_wave"].to(device)
            source_waves = batch["source_waves"].to(device)
            bsz = mix_wave.size(0)

            if model_name in SPEC_MODELS:
                mix_mag = batch["mix_mag"].to(device)
                outputs = model(mix_mag)
                masks = outputs.float().cpu().numpy()
                mix_np = mix_wave.cpu().numpy()
                src_np = source_waves.cpu().numpy()
                th, tw = outputs.size(2), outputs.size(3)
                mix_mag_np = mix_mag[:, :, :th, :tw].float().cpu().numpy()
                for b in range(bsz):
                    for c in range(num_classes):
                        ref = src_np[b, c]
                        if not _active_mask(ref, ref_rms_thresh):
                            continue
                        est = mag_mask_to_waveform(
                            mix_np[b],
                            mix_mag_np[b],
                            masks[b, c],
                            config.n_fft,
                            config.hop_length,
                        )
                        sum_sdr += si_sdr_numpy(est, ref)
                        count_sdr += 1
                        sum_l1 += float(np.mean(np.abs(est - ref)))
                        count_l1 += 1
            else:
                outputs = model(mix_wave)
                est_w = outputs[:, :, : source_waves.size(-1)]
                est_np = est_w.cpu().numpy()
                src_np = source_waves.cpu().numpy()
                for b in range(bsz):
                    for c in range(num_classes):
                        ref = src_np[b, c]
                        if not _active_mask(ref, ref_rms_thresh):
                            continue
                        est = est_np[b, c]
                        sum_sdr += si_sdr_numpy(est, ref)
                        count_sdr += 1
                        sum_l1 += float(np.mean(np.abs(est - ref)))
                        count_l1 += 1

    mean_sdr = sum_sdr / max(1, count_sdr)
    mean_l1 = sum_l1 / max(1, count_l1)
    return {
        "mean_si_sdr_db": mean_sdr,
        "mean_waveform_l1": mean_l1,
        "si_sdr_pairs": float(count_sdr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare separation models via SI-SDR on val split.")
    parser.add_argument("--results-root", default="results", type=str)
    parser.add_argument("--data-dir", default=BALANCED_PRESET.data_dir)
    parser.add_argument("--original-metadata", default=BALANCED_PRESET.original_metadata_path)
    parser.add_argument("--ref-rms-thresh", type=float, default=1e-3, help="Skip class channels quieter than this RMS.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model names; default: all checkpoints found under results-root.",
    )
    args = parser.parse_args()

    config: ExperimentConfig = replace(BALANCED_PRESET, data_dir=args.data_dir, original_metadata_path=args.original_metadata)
    results_root = Path(args.results_root)
    discovered = _discover_checkpoints(results_root)
    if args.models:
        discovered = {k: v for k, v in discovered.items() if k in set(args.models)}
    if not discovered:
        raise SystemExit(f"No checkpoints found under {results_root}. Train models first.")

    class_map = build_class_map()
    audio_cfg = AudioConfig(config.sample_rate, config.samples, config.n_fft, config.hop_length, config.num_classes)
    split = load_split_metadata(config.metadata_path, config.val_ratio, config.test_ratio, config.random_seed)
    val_ds = SeparationDataset(split["val"], config.audio_dir, class_map, config.original_metadata_path, audio_cfg)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = select_device()
    rows: list[dict[str, object]] = []
    for model_name in sorted(discovered.keys()):
        ckpt = discovered[model_name]
        metrics = evaluate_model(model_name, ckpt, config, val_loader, device, args.ref_rms_thresh)
        rows.append(
            {
                "model": model_name,
                "checkpoint": str(ckpt),
                **metrics,
            }
        )

    rows.sort(key=lambda r: float(r["mean_si_sdr_db"]), reverse=True)
    payload = {
        "metric": "mean_si_sdr_db (higher is better; active class channels only)",
        "val_batches": len(val_loader),
        "ranking": rows,
    }
    out_path = results_root / "separation_model_comparison.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    best = rows[0]
    print(f"\nBest model by mean SI-SDR: {best['model']} ({best['mean_si_sdr_db']:.3f} dB)\n")


if __name__ == "__main__":
    main()
