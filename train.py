"""End-to-end CatBoost-only training entry-point.

Usage::

    python train.py

The script:

1. Loads ``train.csv`` and ``test.csv``.
2. Runs the :class:`FeatureBuilder` pipeline.
3. Optimises CatBoost hyperparameters with Optuna.
4. Fits CatBoost with 5-fold CV using the best parameters.
5. Writes ``outputs/submission.csv`` and ``outputs/report.md``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow `python train.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    ARTIFACTS_DIR,
    BEST_PARAMS_JSON,
    REPORT_MD,
    SUBMISSION_CSV,
    TEST_CSV,
    TRAIN_CSV,
    TrainConfig,
    set_global_seed,
)
from src.feature_engineering import FeatureBuilder
from src.reporting import build_report
from src.trainer import run_training
from src.utils import get_logger

logger = get_logger("train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CatBoost demand prediction pipeline.")
    parser.add_argument("--trials", type=int, default=10, help="Optuna trials for CatBoost (20 quick / 100 full)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-mode",
        choices=["groupkfold", "time"],
        default="time",
        help="groupkfold = day-grouped CV, time = expanding past-only validation",
    )
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        seed=args.seed,
        n_folds=args.folds,
        n_optuna_trials=args.trials,
    )
    set_global_seed(cfg.seed)

    t0 = time.time()
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    logger.info("Train=%s  Test=%s", train.shape, test.shape)

    feature_builder = FeatureBuilder(random_state=cfg.seed)
    catboost_result, _train_feat, _test_feat, target_enc = run_training(
        train=train,
        test=test,
        feature_builder=feature_builder,
        target=cfg.target,
        use_log_target=cfg.use_log_target,
        n_folds=cfg.n_folds,
        seed=cfg.seed,
        n_optuna_trials=cfg.n_optuna_trials,
        validation_mode=args.validation_mode,
    )

    # Persist feature builder target-encodings for inference / resuming
    joblib.dump(target_enc, ARTIFACTS_DIR / "target_enc.joblib")
    joblib.dump(feature_builder, ARTIFACTS_DIR / "feature_builder.joblib")

    logger.info("CatBoost mean R² = %.5f ± %.5f", catboost_result["mean_r2"], catboost_result["std_r2"])
    logger.info("Optuna best R² = %.5f across %d trials", catboost_result["search_best_r2"], catboost_result["study_trials"])

    final_pred = np.clip(catboost_result["test_pred"], 0.0, 1.0)
    sub_out = pd.DataFrame(
        {
            "Index": test["Index"].values,
            "demand": final_pred,
        }
    )
    sub_out.to_csv(SUBMISSION_CSV, index=False)
    logger.info("Wrote %s with %d rows", SUBMISSION_CSV, len(sub_out))

    build_report(
        train_result=catboost_result,
        feature_builder=feature_builder,
        test_pred=final_pred,
        report_path=REPORT_MD,
    )
    logger.info("Wrote %s", REPORT_MD)

    if BEST_PARAMS_JSON.exists():
        logger.info("Best params saved to %s", BEST_PARAMS_JSON)

    logger.info("Total wall time: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
