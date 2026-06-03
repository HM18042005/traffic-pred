"""Feature engineering for the demand regression task.

The pipeline is implemented as a :class:`FeatureBuilder` so the same logic
applies to training and inference.  All leakage-sensitive aggregates are
computed inside cross-validation folds by :meth:`fit_transform` /
:meth:`transform`.

The features fall into the following groups:

* Cyclical time encodings for the ``timestamp`` column.
* Spatial features from geohash: latitude, longitude, hierarchical
  prefixes and small-area density.
* Fold-safe geohash-level and geohash/hour statistics (mean, std, count, rank).
* Cross-feature interactions (lanes * road type, etc.).
* Target / out-of-fold target encoding for the high-cardinality
  ``geohash`` and a couple of low-cardinality features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import pygeohash  # type: ignore
except Exception:  # pragma: no cover - pygeohash is a hard dep, but stay robust
    pygeohash = None  # type: ignore


from src.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _parse_timestamp(series: pd.Series) -> pd.DataFrame:
    """Return hour, minute and a 0..1439 minute-of-day integer."""
    split = series.str.split(":", n=1, expand=True)
    hour = split[0].astype(int)
    minute = split[1].astype(int)
    return pd.DataFrame(
        {
            "hour": hour,
            "minute": minute,
            "min_of_day": hour * 60 + minute,
        }
    )


def _add_cyclical(df: pd.DataFrame, col: str, period: int) -> None:
    """In-place add sin/cos columns for a cyclic feature."""
    rad = 2 * np.pi * df[col].astype(float) / period
    df[f"sin_{col}"] = np.sin(rad)
    df[f"cos_{col}"] = np.cos(rad)


def _safe_decode(gh: str) -> tuple[float, float]:
    """Decode geohash to (lat, lon) with a safe fallback."""
    if pygeohash is None:
        return 0.0, 0.0
    try:
        lat, lon = pygeohash.decode(gh)
        return float(lat), float(lon)
    except Exception:
        return 0.0, 0.0


# --------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------
@dataclass
class FeatureBuilder:
    """Stateful feature builder with consistent fit/transform API."""

    target: str = "demand"
    geohash_prefixes: tuple[int, ...] = (5, 4, 3)
    smoothing: float = 30.0  # target-encoding smoothing factor
    n_folds_for_target_enc: int = 5
    random_state: int = 42

    # Populated during fit
    geohash_stats_: Optional[pd.DataFrame] = None
    geohash_hour_stats_: Optional[pd.DataFrame] = None
    global_target_mean_: float = 0.0
    cat_modes_: dict[str, str] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    target_enc_maps_: dict[str, pd.Series] = field(default_factory=dict)
    temperature_fill_value_: float = np.nan
    geohash_lat_lon_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        """Compute statistics that are safe to learn from the full train set.

        These are descriptive aggregates that are not leakage-prone.  Target
        encoding happens inside :meth:`fit_transform` because it must be
        out-of-fold.
        """
        df = train.copy()
        self.global_target_mean_ = float(df[self.target].mean())
        self.temperature_fill_value_ = float(df["Temperature"].median()) if "Temperature" in df.columns else 0.0

        # Categorical modes (used to impute missing values in transform)
        for c in ["RoadType", "Weather", "LargeVehicles", "Landmarks"]:
            if c in df.columns:
                mode = df[c].mode(dropna=True)
                self.cat_modes_[c] = mode.iloc[0] if not mode.empty else "missing"

        # Geohash aggregates (mean/std/count/rank across all training rows)
        agg = df.groupby("geohash")[self.target].agg(["mean", "std", "count", "median"])
        agg.columns = [f"gh_{c}" for c in agg.columns]
        agg["gh_rank"] = agg["gh_mean"].rank(method="dense", ascending=False)
        agg["gh_zscore"] = (agg["gh_mean"] - agg["gh_mean"].mean()) / (agg["gh_mean"].std() + 1e-9)
        self.geohash_stats_ = agg

        # Per (geohash, hour) mean demand - useful for time-of-day local pattern
        ts = _parse_timestamp(df["timestamp"].astype(str))
        df_with_hour = df.assign(hour=ts["hour"].values)
        gh_hour = df_with_hour.groupby(["geohash", "hour"])[self.target].agg(["mean", "count"])
        gh_hour.columns = [f"gh_hour_{c}" for c in gh_hour.columns]
        self.geohash_hour_stats_ = gh_hour

        # Latitude/longitude lookup from geohash strings
        geohashes = pd.Index(df["geohash"].unique())
        lat_lon = pd.DataFrame(
            [_safe_decode(g) for g in geohashes],
            index=geohashes,
            columns=["latitude", "longitude"],
        )
        self.geohash_lat_lon_ = lat_lon
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, target_enc: Optional[dict[str, pd.Series]] = None) -> pd.DataFrame:
        """Apply feature engineering to an arbitrary frame (train or test)."""
        out = df.copy()

        # ---- Time features ----
        ts = _parse_timestamp(out["timestamp"].astype(str))
        out = pd.concat([out, ts], axis=1)
        _add_cyclical(out, "hour", 24)
        _add_cyclical(out, "min_of_day", 24 * 60)

        # ---- Geohash prefixes ----
        for k in self.geohash_prefixes:
            out[f"geohash{k}"] = out["geohash"].astype(str).str[:k]

        # ---- Geohash lat/lon ----
        if self.geohash_lat_lon_ is not None:
            ll = self.geohash_lat_lon_.reindex(out["geohash"].values)
            out["latitude"] = ll["latitude"].values
            out["longitude"] = ll["longitude"].values
        else:
            out["latitude"] = 0.0
            out["longitude"] = 0.0

        out["lat_x_lon"] = out["latitude"] * out["longitude"]

        # ---- Geohash aggregates ----
        if self.geohash_stats_ is not None:
            join = self.geohash_stats_.reindex(out["geohash"].values)
            out["gh_mean"] = join["gh_mean"].fillna(self.global_target_mean_).values
            out["gh_std"] = join["gh_std"].fillna(0.0).values
            out["gh_count"] = join["gh_count"].fillna(0.0).values
            out["gh_median"] = join["gh_median"].fillna(self.global_target_mean_).values
            out["gh_rank"] = join["gh_rank"].fillna(float(len(self.geohash_stats_) + 1)).values
            out["gh_zscore"] = join["gh_zscore"].fillna(0.0).values

        # ---- Per (geohash, hour) aggregates ----
        if self.geohash_hour_stats_ is not None:
            gh_hour = self.geohash_hour_stats_.reindex(
                pd.MultiIndex.from_arrays([out["geohash"].values, out["hour"].values])
            )
            out["gh_hour_mean"] = gh_hour["gh_hour_mean"].fillna(self.global_target_mean_).values
            out["gh_hour_count"] = gh_hour["gh_hour_count"].fillna(0.0).values

        # ---- Categorical imputation ----
        for c, mode in self.cat_modes_.items():
            if c in out.columns:
                out[c] = out[c].fillna(mode)

        # Temperature: impute with the training median only
        if "Temperature" in out.columns:
            out["Temperature"] = out["Temperature"].fillna(self.temperature_fill_value_)

        # ---- Interaction features ----
        out["lane_x_road"] = out["NumberofLanes"].astype(float) * (
            out["RoadType"].map({"Residential": 1.0, "Street": 2.0, "Highway": 3.0}).fillna(1.0)
        )
        out["hour_sin_cos"] = out["sin_hour"] * out["cos_hour"]
        out["is_morning_peak"] = ((out["hour"] >= 7) & (out["hour"] <= 10)).astype(int)
        out["is_evening_peak"] = ((out["hour"] >= 16) & (out["hour"] <= 19)).astype(int)
        out["is_night"] = ((out["hour"] >= 22) | (out["hour"] <= 4)).astype(int)
        out["is_weekend"] = (out["day"] % 7 >= 5).astype(int)

        # ---- Target encoding (provided by fit_transform) ----
        if target_enc:
            for col, mapping in target_enc.items():
                out[f"te_{col}"] = out[col].map(mapping).fillna(self.global_target_mean_)

        return out

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        train: pd.DataFrame,
        folds: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        """Fit the builder and produce out-of-fold target encodings.

        Returns the engineered training frame plus the *full-train* target
        encodings which should be used at inference time.
        """
        self.fit(train)
        df = train.copy()
        engineered = self.transform(df, target_enc=None)

        target_feature_cols = [
            "gh_mean",
            "gh_std",
            "gh_count",
            "gh_median",
            "gh_rank",
            "gh_zscore",
            "gh_hour_mean",
            "gh_hour_count",
        ]

        # Compute OOF target encodings for the high-cardinality geohash
        te_cols = ["geohash", "RoadType", "Weather"]
        # Add prefix-based cols using the strings directly
        for k in self.geohash_prefixes:
            col = f"geohash{k}"
            df[col] = df["geohash"].astype(str).str[:k]
            te_cols.append(col)
        oof_te = {c: np.full(len(df), self.global_target_mean_, dtype=float) for c in te_cols}
        fold_maps: dict[str, list[pd.Series]] = {c: [] for c in te_cols}

        oof_stats = {
            col: np.asarray(engineered[col], dtype=float).copy()
            for col in target_feature_cols
        }

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            tr = df.iloc[tr_idx]
            va = df.iloc[va_idx]
            fold_mean = float(tr[self.target].mean())

            gh_stats = tr.groupby("geohash")[self.target].agg(["mean", "std", "count", "median"])
            gh_stats.columns = [f"gh_{c}" for c in gh_stats.columns]
            gh_stats["gh_rank"] = gh_stats["gh_mean"].rank(method="dense", ascending=False)
            gh_stats["gh_zscore"] = (gh_stats["gh_mean"] - gh_stats["gh_mean"].mean()) / (
                gh_stats["gh_mean"].std() + 1e-9
            )
            gh_join = gh_stats.reindex(va["geohash"].values)
            oof_stats["gh_mean"][va_idx] = gh_join["gh_mean"].fillna(fold_mean).to_numpy(dtype=float)
            oof_stats["gh_std"][va_idx] = gh_join["gh_std"].fillna(0.0).to_numpy(dtype=float)
            oof_stats["gh_count"][va_idx] = gh_join["gh_count"].fillna(0.0).to_numpy(dtype=float)
            oof_stats["gh_median"][va_idx] = gh_join["gh_median"].fillna(fold_mean).to_numpy(dtype=float)
            oof_stats["gh_rank"][va_idx] = gh_join["gh_rank"].fillna(float(len(gh_stats) + 1)).to_numpy(dtype=float)
            oof_stats["gh_zscore"][va_idx] = gh_join["gh_zscore"].fillna(0.0).to_numpy(dtype=float)

            ts_tr = _parse_timestamp(tr["timestamp"].astype(str))
            tr_hour = tr.assign(hour=ts_tr["hour"].values)
            gh_hour = tr_hour.groupby(["geohash", "hour"])[self.target].agg(["mean", "count"])
            gh_hour.columns = [f"gh_hour_{c}" for c in gh_hour.columns]
            va_hour = _parse_timestamp(va["timestamp"].astype(str))["hour"].values
            gh_hour_join = gh_hour.reindex(pd.MultiIndex.from_arrays([va["geohash"].values, va_hour]))
            oof_stats["gh_hour_mean"][va_idx] = gh_hour_join["gh_hour_mean"].fillna(fold_mean).to_numpy(dtype=float)
            oof_stats["gh_hour_count"][va_idx] = gh_hour_join["gh_hour_count"].fillna(0.0).to_numpy(dtype=float)

            for c in te_cols:
                if c not in tr.columns:
                    continue
                stats = tr.groupby(c)[self.target].agg(["mean", "count"])
                # Bayesian smoothing
                smoothed = (
                    (stats["mean"] * stats["count"] + fold_mean * self.smoothing)
                    / (stats["count"] + self.smoothing)
                )
                fold_maps[c].append(smoothed)
                oof_te[c][va_idx] = va[c].map(smoothed).fillna(fold_mean).values

        # Build full-train encodings for use at inference
        target_enc: dict[str, pd.Series] = {}
        for c, maps in fold_maps.items():
            full_stats = df.groupby(c)[self.target].agg(["mean", "count"])
            smoothed = (
                (full_stats["mean"] * full_stats["count"] + self.global_target_mean_ * self.smoothing)
                / (full_stats["count"] + self.smoothing)
            )
            target_enc[c] = smoothed
            self.target_enc_maps_[c] = smoothed

        for col, values in oof_stats.items():
            engineered[col] = values
        for c in te_cols:
            engineered[f"te_{c}"] = oof_te[c]

        # Drop raw columns that are not useful for modelling
        drop = {"Index", self.target, "timestamp"}
        self.feature_names_ = [c for c in engineered.columns if c not in drop]
        return engineered, target_enc

    # ------------------------------------------------------------------
    def feature_columns(self) -> list[str]:
        return list(self.feature_names_)


__all__ = ["FeatureBuilder"]
