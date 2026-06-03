"""CatBoost-only model definitions used by the training pipeline."""
from __future__ import annotations

from typing import Any, Optional


def _default_catboost() -> dict[str, Any]:
    return dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        iterations=4000,
        early_stopping_rounds=200,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        task_type="GPU",
        devices="0",
    )


def get_default(name: str) -> dict[str, Any]:
    """Return a fresh copy of the default hyperparameters for ``name``."""
    if name == "catboost":
        return _default_catboost()
    raise KeyError(name)


class CatBoostWrapper:
    def __init__(self, params: dict[str, Any], cat_features: Optional[list[str]] = None):
        from catboost import CatBoostRegressor  # type: ignore

        self.params = params
        self.cat_features = cat_features or []
        self.model = CatBoostRegressor(**params)

    def fit(self, X, y, X_valid=None, y_valid=None, **kwargs):
        self.model.fit(
            X,
            y,
            eval_set=(X_valid, y_valid) if X_valid is not None else None,
            cat_features=self.cat_features or None,
            use_best_model=X_valid is not None,
            verbose=False,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)


__all__ = ["get_default", "CatBoostWrapper"]
