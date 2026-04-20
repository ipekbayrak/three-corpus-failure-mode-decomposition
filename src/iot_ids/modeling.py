from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = {
    "logistic_regression",
    "extra_trees",
    "hist_gradient_boosting",
    "mlp",
    "ft_transformer",
}

NATURAL_UNIT_INTERVAL_COLUMNS = {
    "src_byte_frac",
    "src_pkt_frac",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "proto_other",
}


class SourceFeatureSanitizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_columns: Iterable[str],
        lower_quantile: float = 0.001,
        upper_quantile: float = 0.999,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> SourceFeatureSanitizer:
        frame = self._to_frame(x)
        sanitized = self._sanitize_base(frame)
        self.feature_names_in_ = list(sanitized.columns)
        self.lower_bounds_ = sanitized.quantile(self.lower_quantile)
        self.upper_bounds_ = sanitized.quantile(self.upper_quantile)
        for column in NATURAL_UNIT_INTERVAL_COLUMNS:
            if column in self.lower_bounds_.index:
                self.lower_bounds_.loc[column] = 0.0
                self.upper_bounds_.loc[column] = 1.0
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = self._to_frame(x)
        sanitized = self._sanitize_base(frame)
        lower = self.lower_bounds_.reindex(sanitized.columns)
        upper = self.upper_bounds_.reindex(sanitized.columns)
        clipped = sanitized.clip(lower=lower, upper=upper, axis=1)
        for column in NATURAL_UNIT_INTERVAL_COLUMNS:
            if column in clipped.columns:
                clipped[column] = clipped[column].clip(0.0, 1.0)
        return clipped.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _to_frame(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.loc[:, self.feature_columns].copy()
        return pd.DataFrame(x, columns=self.feature_columns)

    def _sanitize_base(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "bytes_per_pkt" in numeric.columns:
            numeric["bytes_per_pkt"] = np.log1p(numeric["bytes_per_pkt"].clip(lower=0.0))
        for column in NATURAL_UNIT_INTERVAL_COLUMNS:
            if column in numeric.columns:
                numeric[column] = numeric[column].clip(0.0, 1.0)
        return numeric


class BoundedFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_columns: Iterable[str],
        clip_value: float = 8.0,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.clip_value = clip_value

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> BoundedFeatureScaler:
        frame = self._to_frame(x)
        self.scale_columns_ = [
            column
            for column in self.feature_columns
            if column not in NATURAL_UNIT_INTERVAL_COLUMNS
        ]
        self.passthrough_columns_ = [
            column for column in self.feature_columns if column in NATURAL_UNIT_INTERVAL_COLUMNS
        ]
        self.scaler_ = StandardScaler()
        if self.scale_columns_:
            self.scaler_.fit(frame[self.scale_columns_])
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = self._to_frame(x)
        transformed = frame.copy()
        if self.scale_columns_:
            scaled = self.scaler_.transform(frame[self.scale_columns_])
            transformed.loc[:, self.scale_columns_] = np.clip(
                scaled,
                -self.clip_value,
                self.clip_value,
            )
        for column in self.passthrough_columns_:
            transformed[column] = transformed[column].clip(0.0, 1.0)
        return transformed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _to_frame(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.loc[:, self.feature_columns].copy()
        return pd.DataFrame(x, columns=self.feature_columns)


def make_model_core(
    model_name: str,
    seed: int,
    feature_columns: Iterable[str] | None = None,
) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.1,
            max_iter=100,
            random_state=seed,
        )
    if model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            solver="lbfgs",
            alpha=1e-3,
            max_iter=80,
            random_state=seed,
        )
    if model_name == "ft_transformer":
        if feature_columns is None:
            raise ValueError("feature_columns are required for ft_transformer")
        from iot_ids.ft_transformer import FTTransformerEstimator

        return FTTransformerEstimator(feature_columns=feature_columns, seed=seed)
    raise ValueError(f"Unsupported model name: {model_name}")


def make_estimator(model_name: str, feature_columns: Iterable[str], seed: int) -> Any:
    sanitizer = SourceFeatureSanitizer(feature_columns)
    scaler = BoundedFeatureScaler(feature_columns)
    estimator = make_model_core(model_name, seed, feature_columns)
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("sanitize", sanitizer),
                ("scale", scaler),
                ("model", estimator),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline([("sanitize", sanitizer), ("model", estimator)])
    if model_name == "hist_gradient_boosting":
        return Pipeline([("sanitize", sanitizer), ("model", estimator)])
    if model_name == "mlp":
        return Pipeline(
            [
                ("sanitize", sanitizer),
                ("scale", scaler),
                ("model", estimator),
            ]
        )
    if model_name == "ft_transformer":
        return estimator
    raise ValueError(f"Unsupported model name: {model_name}")


def fit_estimator(
    model_name: str,
    train_frame: pd.DataFrame,
    feature_columns: Iterable[str],
    seed: int,
) -> Any:
    feature_columns = list(feature_columns)
    estimator = make_estimator(model_name, feature_columns, seed)
    fit_x = train_frame[list(feature_columns)]
    fit_y = train_frame["label"]
    if model_name == "ft_transformer":
        groups = train_frame["group_id"] if "group_id" in train_frame.columns else None
        estimator.fit(fit_x, fit_y, groups=groups)
        return estimator
    if model_name in {"mlp", "logistic_regression"}:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"sklearn\.utils\.extmath",
            )
            estimator.fit(fit_x, fit_y)
    else:
        estimator.fit(fit_x, fit_y)
    return estimator


def score_positive_class(
    estimator: Any,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
) -> np.ndarray:
    features = frame[list(feature_columns)]
    final_model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else None
    suppress_extmath_warning = isinstance(final_model, (MLPClassifier, LogisticRegression))
    if hasattr(estimator, "predict_proba"):
        if suppress_extmath_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r"sklearn\.utils\.extmath",
                )
                return estimator.predict_proba(features)[:, 1]
        return estimator.predict_proba(features)[:, 1]
    if hasattr(estimator, "decision_function"):
        if suppress_extmath_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r"sklearn\.utils\.extmath",
                )
                raw = estimator.decision_function(features)
        else:
            raw = estimator.decision_function(features)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -60.0, 60.0)))
    raise TypeError("Estimator does not support probability or decision scores")
