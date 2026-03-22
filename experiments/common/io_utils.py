"""Filesystem and run-artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_run_dirs(output_root: str, model_name: str, preset_name: str) -> dict[str, Path]:
    run_id = f"{timestamp_id()}_{model_name}_{preset_name}"
    root = Path(output_root) / run_id
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "metrics": root / "metrics",
        "plots": root / "plots",
        "audio_examples": root / "audio_examples",
        "logs": root / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)


def save_config(path: Path, config_obj: Any) -> None:
    try:
        payload = asdict(config_obj)
    except TypeError:
        payload = dict(config_obj)
    write_json(path, payload)

