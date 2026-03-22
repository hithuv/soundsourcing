"""Default experiment presets for local M3 Pro training."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    data_dir: str = "UrbanSound8K_separation"
    original_metadata_path: str = "UrbanSound8K/metadata/UrbanSound8K.csv"
    output_root: str = "results"
    sample_rate: int = 22050
    duration_sec: int = 4
    n_fft: int = 1024
    hop_length: int = 256
    num_classes: int = 10
    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    early_stopping_patience: int = 6
    save_audio_examples: int = 12

    @property
    def samples(self) -> int:
        return self.sample_rate * self.duration_sec

    @property
    def metadata_path(self) -> Path:
        return Path(self.data_dir) / "metadata.csv"

    @property
    def audio_dir(self) -> Path:
        return Path(self.data_dir) / "audio"


BALANCED_PRESET = ExperimentConfig(run_name="balanced")

