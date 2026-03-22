"""Shared dataset and split logic for separation experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    samples: int
    n_fft: int
    hop_length: int
    num_classes: int


def build_class_map(label_encoder_path: str = "output/label_encoder.npy") -> dict[str, int]:
    classes = np.load(label_encoder_path, allow_pickle=True)
    return {name: index for index, name in enumerate(classes)}


def load_split_metadata(
    metadata_path: Path,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(metadata_path)
    train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=random_seed)
    adjusted_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=adjusted_test_ratio, random_state=random_seed)
    return {"train": train_df.reset_index(drop=True), "val": val_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


def _fix_length(wave: np.ndarray, target_samples: int) -> np.ndarray:
    if len(wave) < target_samples:
        return np.pad(wave, (0, target_samples - len(wave)))
    return wave[:target_samples]


def _to_mag(wave: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    stft = librosa.stft(wave, n_fft=n_fft, hop_length=hop_length)
    mag, _phase = librosa.magphase(stft)
    return mag


class SeparationDataset(Dataset):
    """Returns both waveform and spectrogram targets for shared training code."""

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        class_map: dict[str, int],
        original_metadata_path: str,
        audio_cfg: AudioConfig,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.df = df
        self.audio_dir = audio_dir
        self.class_map = class_map
        self.audio_cfg = audio_cfg
        self.augment_fn = augment_fn
        self.original_metadata_df = pd.read_csv(original_metadata_path)
        self.filename_to_class = {
            row["slice_file_name"]: row["class"] for _, row in self.original_metadata_df.iterrows()
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        mix_path = self.audio_dir / row["mix_filename"]
        mix_wave, _ = librosa.load(mix_path, sr=self.audio_cfg.sample_rate)
        mix_wave = _fix_length(mix_wave, self.audio_cfg.samples)
        if self.augment_fn is not None:
            mix_wave = self.augment_fn(mix_wave)

        source_waves = np.zeros((self.audio_cfg.num_classes, self.audio_cfg.samples), dtype=np.float32)
        for source_path in row["source_paths"].split(";"):
            source_wave, _ = librosa.load(source_path, sr=self.audio_cfg.sample_rate)
            source_wave = _fix_length(source_wave, self.audio_cfg.samples)
            source_name = os.path.basename(source_path)
            class_name = self.filename_to_class[source_name]
            source_waves[self.class_map[class_name]] += source_wave

        mix_mag = _to_mag(mix_wave, self.audio_cfg.n_fft, self.audio_cfg.hop_length).astype(np.float32)
        target_mags = np.zeros((self.audio_cfg.num_classes, mix_mag.shape[0], mix_mag.shape[1]), dtype=np.float32)
        for class_idx in range(self.audio_cfg.num_classes):
            target_mags[class_idx] = _to_mag(
                source_waves[class_idx], self.audio_cfg.n_fft, self.audio_cfg.hop_length
            ).astype(np.float32)
            if target_mags[class_idx].shape[1] != mix_mag.shape[1]:
                target_mags[class_idx] = librosa.util.fix_length(
                    target_mags[class_idx], size=mix_mag.shape[1], axis=1
                )

        return {
            "mix_wave": torch.tensor(mix_wave, dtype=torch.float32),
            "source_waves": torch.tensor(source_waves, dtype=torch.float32),
            "mix_mag": torch.tensor(mix_mag, dtype=torch.float32).unsqueeze(0),
            "target_mags": torch.tensor(target_mags, dtype=torch.float32),
            "mix_filename": row["mix_filename"],
        }

