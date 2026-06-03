"""Configuration and constants for the demand-prediction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

TRAIN_CSV: Path = DATA_DIR / "train.csv"
TEST_CSV: Path = DATA_DIR / "test.csv"
SAMPLE_SUB_CSV: Path = DATA_DIR / "sample_submission.csv"

SUBMISSION_CSV: Path = OUTPUTS_DIR / "submission.csv"
REPORT_MD: Path = OUTPUTS_DIR / "report.md"

# Persisted training artefacts
OOF_DIR: Path = ARTIFACTS_DIR / "oof"
TEST_PRED_DIR: Path = ARTIFACTS_DIR / "test_preds"
BEST_PARAMS_JSON: Path = ARTIFACTS_DIR / "best_params.json"
OPTUNA_STUDY_PKL: Path = ARTIFACTS_DIR / "optuna_study.pkl"
OOF_DIR.mkdir(parents=True, exist_ok=True)
TEST_PRED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Set seeds for python, numpy, and torch (if present) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Modelling defaults
# --------------------------------------------------------------------------------------
TARGET: str = "demand"
ID_COL: str = "Index"
N_FOLDS: int = 5

CATEGORICAL_COLS: list[str] = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "geohash5",
    "geohash4",
    "geohash3",
]

NUMERIC_BASE: list[str] = [
    "day",
    "NumberofLanes",
    "Temperature",
    "hour",
    "minute",
    "min_of_day",
    "sin_hour",
    "cos_hour",
    "sin_min",
    "cos_min",
    "latitude",
    "longitude",
    "lat_x_lon",
    "lane_x_road",
]


@dataclass
class TrainConfig:
    """High-level training configuration."""

    seed: int = SEED
    n_folds: int = N_FOLDS
    target: str = TARGET
    n_optuna_trials: int = 20
    use_log_target: bool = True  # log1p target because the distribution is heavy-tailed
