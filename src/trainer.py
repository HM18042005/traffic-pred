"""CatBoost-only training orchestrator.

The pipeline keeps the strongest feature engineering and cross-validation
logic, but removes all multi-model branches, ensembling, stacking, and
blending.
"""
from __future__ import annotations

import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.config import (
    BEST_PARAMS_JSON,
    N_FOLDS,
    OOF_DIR,
    OPTUNA_STUDY_PKL,
    SEED,
    TEST_PRED_DIR,
    set_global_seed,
)
from src.cv import make_groupkfold_by_day, make_time_based_folds, r2_score_np
from src.feature_engineering import FeatureBuilder
from src.models import CatBoostWrapper, get_default
from src.utils import get_logger

logger = get_logger(__name__)

CATBOOST_CATEGORICAL_COLS = {
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "geohash5",
    "geohash4",
    "geohash3",
}


# --------------------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------------------
def _save_best_params(best_params: dict[str, Any]) -> None:
    BEST_PARAMS_JSON.write_text(json.dumps(best_params, indent=2, sort_keys=True), encoding="utf-8")


def _save_study(study) -> None:
    joblib.dump(study, OPTUNA_STUDY_PKL)


# --------------------------------------------------------------------------------------
# CatBoost CV fit
# --------------------------------------------------------------------------------------
def _train_catboost(
    params: dict[str, Any],
    train_feat: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    test_feat: pd.DataFrame,
    cat_features: list[str],
    target_in_log: bool,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    logger.info(
        "  [catboost] final fit on %d train rows × %d test rows",
        len(train_feat),
        len(test_feat),
    )
    t_model = time.perf_counter()
    oof = np.zeros(len(train_feat), dtype=float)
    test_preds = np.zeros(len(test_feat), dtype=float)
    fold_scores: list[float] = []

    feat_cols = list(train_feat.columns)
    train_feat = train_feat[feat_cols].copy()
    test_feat = test_feat[feat_cols].copy()

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        t_fold = time.perf_counter()
        Xtr = train_feat.iloc[tr_idx].copy()
        ytr = y[tr_idx]
        Xva = train_feat.iloc[va_idx].copy()
        yva = y[va_idx]

        model = CatBoostWrapper(params, cat_features=cat_features)
        logger.info(
            "  [catboost] fold %d/%d fitting on %d train / %d val ...",
            fold_idx,
            len(folds),
            len(Xtr),
            len(Xva),
        )
        model.fit(Xtr, ytr, X_valid=Xva, y_valid=yva)
        va_pred = np.asarray(model.predict(Xva), dtype=float)
        te_pred = np.asarray(model.predict(test_feat), dtype=float)

        oof[va_idx] = va_pred
        test_preds += te_pred

        y_true = np.expm1(yva) if target_in_log else yva
        y_pred = np.expm1(va_pred) if target_in_log else va_pred
        score = r2_score_np(y_true, y_pred)
        fold_scores.append(score)
        logger.info(
            "  [catboost] fold %d/%d R² = %.5f  (%.1fs)",
            fold_idx,
            len(folds),
            score,
            time.perf_counter() - t_fold,
        )

    test_preds /= max(len(folds), 1)
    if target_in_log:
        oof_orig = np.expm1(oof)
        test_orig = np.expm1(test_preds)
    else:
        oof_orig = oof
        test_orig = test_preds

    mean = float(np.mean(fold_scores))
    logger.info("  [catboost] final mean R² = %.5f  (%.1fs)", mean, time.perf_counter() - t_model)
    return oof_orig, test_orig, fold_scores, mean


# --------------------------------------------------------------------------------------
# Optuna search
# --------------------------------------------------------------------------------------
def _optuna_search(
    n_trials: int,
    train_feat: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cat_features: list[str],
    target_in_log: bool,
    seed: int,
) -> tuple[dict[str, Any], float, Any]:
    import optuna  # type: ignore
    from optuna.samplers import TPESampler  # type: ignore

    base = get_default("catboost")
    feat_cols = list(train_feat.columns)
    train_feat = train_feat[feat_cols].copy()
    trial_folds = folds[: min(3, len(folds))]

    study = None
    if OPTUNA_STUDY_PKL.exists():
        try:
            study = joblib.load(OPTUNA_STUDY_PKL)
            logger.info("Resuming Optuna study from %s (%d completed trials)", OPTUNA_STUDY_PKL, len(study.trials))
        except Exception as exc:  # pragma: no cover - defensive recovery path
            logger.warning("Could not load existing Optuna study: %s", exc)

    if study is None:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=seed, multivariate=True),
        )
        logger.info("Created a new Optuna study")

    completed = len(study.trials)
    remaining_trials = max(n_trials - completed, 0)
    if remaining_trials == 0:
        logger.info("Optuna already has %d trials; skipping search", completed)
        return dict(study.best_params), float(study.best_value), study

    def objective(trial):
        trial_no = trial.number + 1
        trial_start = time.perf_counter()
        logger.info("[catboost] trial %d/%d started", trial_no, n_trials)

        params = dict(base)
        params.update(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 5, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            iterations=trial.suggest_int("iterations", 1500, 6000),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count=trial.suggest_int("border_count", 32, 254),
        )

        scores: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(trial_folds, start=1):
            Xtr = train_feat.iloc[tr_idx].copy()
            ytr = y[tr_idx]
            Xva = train_feat.iloc[va_idx].copy()
            yva = y[va_idx]

            model = CatBoostWrapper(params, cat_features=cat_features)
            model.fit(Xtr, ytr, X_valid=Xva, y_valid=yva)
            pred = np.asarray(model.predict(Xva), dtype=float)
            if target_in_log:
                pred = np.expm1(pred)
                yt = np.expm1(yva)
            else:
                yt = yva
            scores.append(r2_score(yt, pred))
            logger.info(
                "[catboost] trial %d/%d fold %d/%d R² = %.5f",
                trial_no,
                n_trials,
                fold_idx,
                len(trial_folds),
                scores[-1],
            )

        elapsed = time.perf_counter() - trial_start
        trial.set_user_attr("elapsed_s", elapsed)
        score = float(np.mean(scores))
        logger.info("[catboost] trial %d/%d finished in %.1fs with mean R² = %.5f", trial_no, n_trials, elapsed, score)
        return score

    def on_trial_complete(study, trial) -> None:
        _save_study(study)
        trial_no = trial.number + 1
        best_score = float(study.best_value)
        logger.info(
            "[catboost] trial %d/%d complete: value=%.5f best_so_far=%.5f elapsed=%.1fs",
            trial_no,
            n_trials,
            trial.value if trial.value is not None else float("nan"),
            best_score,
            float(trial.user_attrs.get("elapsed_s", 0.0)),
        )
        if study.best_trial.number == trial.number:
            logger.info("[catboost] new best parameters found at trial %d", trial_no)
            _save_best_params(dict(study.best_params))

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=remaining_trials, callbacks=[on_trial_complete])
    total_elapsed = time.perf_counter() - t0
    logger.info(
        "Optuna best CatBoost R² = %.5f in %.0fs (%d total trials)",
        float(study.best_value),
        total_elapsed,
        len(study.trials),
    )
    _save_best_params(dict(study.best_params))
    _save_study(study)
    return dict(study.best_params), float(study.best_value), study


# --------------------------------------------------------------------------------------
# Public training entry point
# --------------------------------------------------------------------------------------
def run_training(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_builder: FeatureBuilder,
    target: str = "demand",
    use_log_target: bool = True,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    n_optuna_trials: int = 20,
    validation_mode: str = "time",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """Train CatBoost only and return the model artefacts."""
    set_global_seed(seed)
    if validation_mode == "groupkfold":
        folds = make_groupkfold_by_day(train, n_folds=n_folds)
    elif validation_mode == "time":
        folds = make_time_based_folds(train, n_folds=n_folds)
    else:
        raise ValueError(f"Unknown validation mode: {validation_mode}")

    train_feat, target_enc = feature_builder.fit_transform(train, folds)
    test_feat = feature_builder.transform(test, target_enc=target_enc)

    feat_cols = feature_builder.feature_columns()
    train_feat = train_feat[feat_cols].copy()
    test_feat = test_feat[feat_cols].copy()

    if use_log_target:
        y = np.log1p(train[target].values)
    else:
        y = train[target].values.astype(float)

    keep = ~np.isnan(y)
    train_feat = train_feat.loc[keep].reset_index(drop=True)
    y = y[keep]

    cat_features = [c for c in feat_cols if c in CATBOOST_CATEGORICAL_COLS and c in train_feat.columns]
    logger.info("CatBoost categorical columns: %s", ", ".join(cat_features) if cat_features else "<none>")

    logger.info("== Optuna search for CatBoost (%d trials requested) ==", n_optuna_trials)
    best_params, best_search_score, study = _optuna_search(
        n_trials=n_optuna_trials,
        train_feat=train_feat,
        y=y,
        folds=folds,
        cat_features=cat_features,
        target_in_log=use_log_target,
        seed=seed,
    )

    logger.info("Best Optuna CatBoost R² = %.5f", best_search_score)
    oof, test_pred, fold_scores, mean = _train_catboost(
        params=best_params,
        train_feat=train_feat,
        y=y,
        folds=folds,
        test_feat=test_feat,
        cat_features=cat_features,
        target_in_log=use_log_target,
    )

    np.save(OOF_DIR / "catboost_oof.npy", oof)
    np.save(TEST_PRED_DIR / "catboost_test.npy", test_pred)

    result = {
        "best_params": best_params,
        "search_best_r2": best_search_score,
        "final_oof_r2": mean,
        "oof": oof,
        "test_pred": test_pred,
        "fold_scores": fold_scores,
        "mean_r2": mean,
        "std_r2": float(np.std(fold_scores)),
        "study_trials": len(study.trials),
        "validation_mode": validation_mode,
    }
    return result, train_feat, test_feat, target_enc


__all__ = ["run_training"]
