"""CatBoost-only inference / re-scoring entry-point.

The script loads persisted artefacts (best hyperparameters, target-encoding
maps, feature builder state) and produces ``outputs/submission.csv`` from
``dataset/test.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    ARTIFACTS_DIR,
    SEED,
    SUBMISSION_CSV,
    TEST_CSV,
    TEST_PRED_DIR,
    TRAIN_CSV,
    BEST_PARAMS_JSON,
    set_global_seed,
)
from src.utils import get_logger

logger = get_logger("inference")


def _predict_with_persisted_catboost(
    test_feat: pd.DataFrame,
    feature_builder,
    target_enc,
) -> dict[str, np.ndarray]:
    """Re-fit CatBoost on the full training data with persisted hyperparameters."""
    from src.models import CatBoostWrapper

    train = pd.read_csv(TRAIN_CSV)
    y = np.log1p(train["demand"].values)
    train_feat = feature_builder.transform(train, target_enc=target_enc)
    feat_cols = feature_builder.feature_columns()
    train_feat = train_feat[feat_cols].copy()
    test_feat = test_feat[feat_cols].copy()

    best_params = json.loads(BEST_PARAMS_JSON.read_text())
    model = CatBoostWrapper(best_params, cat_features=[c for c in feat_cols if c in {"geohash", "RoadType", "LargeVehicles", "Landmarks", "Weather", "geohash5", "geohash4", "geohash3"}])
    model.fit(train_feat, y)
    return {"catboost": np.expm1(np.asarray(model.predict(test_feat), dtype=float))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force re-fit and re-predict")
    args = parser.parse_args(argv)

    set_global_seed(SEED)
    test = pd.read_csv(TEST_CSV)

    feature_builder: object = joblib.load(ARTIFACTS_DIR / "feature_builder.joblib")
    target_enc = joblib.load(ARTIFACTS_DIR / "target_enc.joblib")
    test_feat = feature_builder.transform(test, target_enc=target_enc)

    test_preds: dict[str, np.ndarray] = {}
    if args.rebuild:
        logger.info("Rebuild requested - re-fitting CatBoost from scratch")
        test_preds = _predict_with_persisted_catboost(test_feat, feature_builder, target_enc)
        for n, p in test_preds.items():
            np.save(TEST_PRED_DIR / f"{n}_test.npy", p)
    else:
        path = TEST_PRED_DIR / "catboost_test.npy"
        if path.exists():
            test_preds["catboost"] = np.load(path)

    if not test_preds:
        logger.error("No CatBoost test predictions found - run training first or pass --rebuild")
        return 1

    final = np.clip(test_preds["catboost"], 0.0, 1.0)

    sub_out = pd.DataFrame({"Index": test["Index"].values, "demand": final})
    sub_out.to_csv(SUBMISSION_CSV, index=False)
    logger.info(f"Wrote {SUBMISSION_CSV} ({len(sub_out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
