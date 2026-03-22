"""Unified training utilities for separation models."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.common.io_utils import prepare_run_dirs, save_config, write_json
from experiments.configs.defaults import BALANCED_PRESET, ExperimentConfig
from experiments.data.separation_dataset import AudioConfig, SeparationDataset, build_class_map, load_split_metadata
from experiments.models.baseline_unet import BaselineUNet
from experiments.models.conv_tasnet import ConvTasNetLite
from experiments.models.demucs_lite import DemucsLite
from experiments.models.resattn_unet import ResidualAttentionUNet
from experiments.models.tfctdf_unet import TFCTDFUNet


MODEL_FACTORIES = {
    "baseline_unet": lambda num_classes: BaselineUNet(num_classes),
    "resattn_unet": lambda num_classes: ResidualAttentionUNet(num_classes),
    "tfctdf_unet": lambda num_classes: TFCTDFUNet(num_classes),
    "conv_tasnet": lambda num_classes: ConvTasNetLite(num_classes=num_classes),
    "demucs_lite": lambda num_classes: DemucsLite(num_classes=num_classes),
}
SPEC_MODELS = {"baseline_unet", "resattn_unet", "tfctdf_unet"}


def parse_args(default_model: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train separation model benchmark.")
    parser.add_argument("--model", default=default_model, choices=MODEL_FACTORIES.keys())
    parser.add_argument("--epochs", type=int, default=BALANCED_PRESET.epochs)
    parser.add_argument("--batch-size", type=int, default=BALANCED_PRESET.batch_size)
    parser.add_argument("--lr", type=float, default=BALANCED_PRESET.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=BALANCED_PRESET.weight_decay)
    parser.add_argument("--seed", type=int, default=BALANCED_PRESET.random_seed)
    parser.add_argument("--output-root", default=BALANCED_PRESET.output_root)
    parser.add_argument("--data-dir", default=BALANCED_PRESET.data_dir)
    parser.add_argument("--original-metadata", default=BALANCED_PRESET.original_metadata_path)
    parser.add_argument("--num-workers", type=int, default=BALANCED_PRESET.num_workers)
    parser.add_argument("--early-stopping-patience", type=int, default=BALANCED_PRESET.early_stopping_patience)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_loss(model_name: str, outputs: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if model_name in SPEC_MODELS:
        target_h, target_w = outputs.size(2), outputs.size(3)
        mix_mag = batch["mix_mag"][:, :, :target_h, :target_w]
        target = batch["target_mags"][:, :, :target_h, :target_w]
        reconstructed = mix_mag * outputs
        l1_loss = torch.mean(torch.abs(reconstructed - target))
        ratio_target = target / (mix_mag + 1e-6)
        ratio_loss = torch.mean(torch.abs(outputs - torch.clamp(ratio_target, 0.0, 1.0)))
        return 0.7 * l1_loss + 0.3 * ratio_loss
    target_wave = batch["source_waves"]
    outputs = outputs[:, :, : target_wave.size(-1)]
    return torch.mean(torch.abs(outputs - target_wave))


def run_training(default_model: str) -> None:
    args = parse_args(default_model=default_model)
    seed_everything(args.seed)
    device = select_device()
    config: ExperimentConfig = replace(
        BALANCED_PRESET,
        run_name="balanced",
        data_dir=args.data_dir,
        original_metadata_path=args.original_metadata,
        output_root=args.output_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        random_seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
    )

    run_dirs = prepare_run_dirs(config.output_root, args.model, config.run_name)
    save_config(run_dirs["root"] / "config.json", config)
    write_json(run_dirs["root"] / "run_info.json", {"model": args.model, "device": str(device)})

    class_map = build_class_map()
    audio_cfg = AudioConfig(config.sample_rate, config.samples, config.n_fft, config.hop_length, config.num_classes)
    split = load_split_metadata(config.metadata_path, config.val_ratio, config.test_ratio, config.random_seed)
    train_ds = SeparationDataset(split["train"], config.audio_dir, class_map, config.original_metadata_path, audio_cfg)
    val_ds = SeparationDataset(split["val"], config.audio_dir, class_map, config.original_metadata_path, audio_cfg)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = MODEL_FACTORIES[args.model](config.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, float]] = []
    start = time.time()

    def persist_epoch_csv() -> None:
        if history:
            pd.DataFrame(history).to_csv(run_dirs["metrics"] / "train_history.csv", index=False)

    def finalize_training() -> None:
        torch.save(model.state_dict(), run_dirs["checkpoints"] / "last.pth")
        persist_epoch_csv()
        summary = {
            "model": args.model,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "elapsed_sec": time.time() - start,
            "device": str(device),
            "run_dir": str(run_dirs["root"]),
        }
        write_json(run_dirs["metrics"] / "train_summary.json", summary)
        print(json.dumps(summary, indent=2))

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            train_loss = 0.0
            train_steps = 0
            for batch in tqdm(train_loader, desc=f"{args.model} {epoch}/{config.epochs} train"):
                optimizer.zero_grad()
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch["mix_mag"]) if args.model in SPEC_MODELS else model(batch["mix_wave"])
                loss = compute_loss(args.model, outputs, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += float(loss.item())
                train_steps += 1

            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"{args.model} {epoch}/{config.epochs} val"):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs = model(batch["mix_mag"]) if args.model in SPEC_MODELS else model(batch["mix_wave"])
                    loss = compute_loss(args.model, outputs, batch)
                    val_loss += float(loss.item())
                    val_steps += 1

            scheduler.step()
            avg_train = train_loss / max(1, train_steps)
            avg_val = val_loss / max(1, val_steps)
            lr = float(optimizer.param_groups[0]["lr"])
            history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "lr": lr})
            print(f"epoch={epoch} train_loss={avg_train:.5f} val_loss={avg_val:.5f} lr={lr:.6f}")
            persist_epoch_csv()

            if avg_val < best_val:
                best_val = avg_val
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), run_dirs["checkpoints"] / "best.pth")
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
    finally:
        if history:
            finalize_training()


if __name__ == "__main__":
    run_training(default_model="baseline_unet")

