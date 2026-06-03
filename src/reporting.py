"""Reporting helpers for the CatBoost-only pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def build_report(
    train_result: Mapping[str, Any],
    feature_builder: Any,
    test_pred: np.ndarray,
    report_path: Path,
) -> None:
    """Write a concise markdown report for the CatBoost-only run."""
    lines: list[str] = []
    lines.append("# Demand Prediction - CatBoost Report\n")
    lines.append(
        "This report summarises the single-model pipeline used to predict the "
        "`demand` column. Validation scores are 5-fold out-of-fold (OOF) R².\n"
    )

    lines.append("## 1. Final model summary\n")
    lines.append(f"- Final OOF R²: **{train_result['final_oof_r2']:.5f}**")
    lines.append(f"- OOF mean R²: **{train_result['mean_r2']:.5f}**")
    lines.append(f"- OOF std R²: **{train_result['std_r2']:.5f}**")
    lines.append(f"- Optuna best R²: **{train_result['search_best_r2']:.5f}**")
    lines.append(f"- Optuna trials completed: **{train_result['study_trials']}**\n")

    lines.append("## 2. Best parameters\n")
    params = train_result.get("best_params", {})
    for key in sorted(params):
        lines.append(f"- {key}: `{_short(params[key])}`")
    lines.append("")

    lines.append("## 3. Fold scores\n")
    lines.append(", ".join(f"{score:.5f}" for score in train_result.get("fold_scores", [])) + "\n")

    lines.append("## 4. Feature engineering\n")
    lines.append(
        "The retained feature set includes timestamp-derived cyclical features, "
        "geohash prefix hierarchy, geohash latitude/longitude decoding, "
        "geohash statistics, per-(geohash, hour) statistics, day-48 lookup features, "
        "interaction features, and out-of-fold target encoding for geohash and the "
        "other high-signal categorical fields.\n"
    )
    lines.append(f"- Engineered feature count: **{len(feature_builder.feature_columns())}**\n")

    lines.append("## 5. Submission\n")
    lines.append(
        f"`outputs/submission.csv` contains clipped predictions for the full test set ({len(test_pred)} rows).\n"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) < 32 else s[:29] + "..."
