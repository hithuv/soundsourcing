import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.train.train_utils import run_training


if __name__ == "__main__":
    run_training(default_model="resattn_unet")

