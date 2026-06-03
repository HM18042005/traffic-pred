"""Cross-validation utilities."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from src.config import N_FOLDS, SEED


def make_kfolds(n_rows: int, n_folds: int = N_FOLDS, seed: int = SEED) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Legacy shuffled KFold helper retained for backwards compatibility."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_rows)))


def make_groupkfold_by_day(train: pd.DataFrame, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split rows so that all samples from the same day stay together."""
    groups = train["day"].to_numpy()
    splitter = GroupKFold(n_splits=n_folds)
    return list(splitter.split(np.arange(len(train)), groups=groups))


def make_time_based_folds(train: pd.DataFrame, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window folds with strictly past-only training days."""
    days = np.array(sorted(pd.Index(train["day"].dropna().unique())))
    if len(days) < 2:
        raise ValueError("Need at least two unique days for time-based validation")

    blocks = np.array_split(days, n_folds + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(blocks)):
        train_days = np.concatenate(blocks[:i])
        valid_days = blocks[i]
        tr_idx = np.flatnonzero(train["day"].isin(train_days).to_numpy())
        va_idx = np.flatnonzero(train["day"].isin(valid_days).to_numpy())
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        folds.append((tr_idx, va_idx))
    return folds


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, computed in numpy (robust to constant input)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


__all__ = ["make_kfolds", "make_groupkfold_by_day", "make_time_based_folds", "r2_score_np"]
