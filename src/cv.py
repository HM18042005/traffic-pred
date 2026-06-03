"""Cross-validation utilities."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import N_FOLDS, SEED


def make_kfolds(n_rows: int, n_folds: int = N_FOLDS, seed: int = SEED) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, valid_idx)`` tuples from a KFold splitter."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_rows)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, computed in numpy (robust to constant input)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


__all__ = ["make_kfolds", "r2_score_np"]
