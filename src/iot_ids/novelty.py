from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from iot_ids.modeling import BoundedFeatureScaler, SourceFeatureSanitizer


def compute_oracle_alpha(
    source_attack_groups: Iterable[str],
    target_attack_groups: Iterable[str],
    excluded_groups: Iterable[str] = ("other",),
) -> float:
    excluded = {str(group) for group in excluded_groups}
    source_groups = {str(group) for group in source_attack_groups if str(group) not in excluded}
    target_groups = [str(group) for group in target_attack_groups if str(group) not in excluded]
    if not target_groups:
        return 0.0
    unseen = sum(group not in source_groups for group in target_groups)
    return float(unseen / len(target_groups))


def mapping_coverage(
    attack_groups: Iterable[str],
    excluded_group: str = "other",
) -> tuple[float, float]:
    groups = [str(group) for group in attack_groups]
    if not groups:
        return 1.0, 0.0
    other_share = float(sum(group == excluded_group for group in groups) / len(groups))
    return float(1.0 - other_share), other_share


@dataclass
class NoveltySummary:
    alpha_hat: float
    selected_count: int
    selected_by_fallback: bool
    cutoff_distance: float


class KNNNoveltyProxy:
    def __init__(self, k_nn: int = 10, distance_quantile: float = 0.95) -> None:
        self.k_nn = k_nn
        self.distance_quantile = distance_quantile
        self.nn: NearestNeighbors | None = None
        self.cutoff_distance: float | None = None

    def fit(self, source_attack_features: np.ndarray) -> KNNNoveltyProxy:
        if source_attack_features.shape[0] < 2:
            raise ValueError("Need at least two source attack samples for k-NN novelty scoring")
        effective_k = min(self.k_nn, source_attack_features.shape[0] - 1)
        self.nn = NearestNeighbors(n_neighbors=effective_k + 1)
        self.nn.fit(source_attack_features)
        distances, _ = self.nn.kneighbors(source_attack_features)
        loo_distances = distances[:, -1]
        self.cutoff_distance = float(np.quantile(loo_distances, self.distance_quantile))
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if self.nn is None:
            raise RuntimeError("Novelty proxy must be fit before scoring")
        distances, _ = self.nn.kneighbors(features)
        return distances[:, -1]

    def summarize(
        self,
        features: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        fallback_k: int,
        fallback_strategy: str = "top_k",
        fallback_seed: int = 42,
    ) -> NoveltySummary:
        """Compute alpha_hat novelty summary on selected target instances.

        Parameters
        ----------
        fallback_strategy : str
            How to select instances when zero predictions are positive.
            "top_k" selects the K highest-scoring instances (default, original).
            "random_sample" selects K instances uniformly at random, avoiding
            coupling to the detector's confidence calibration.
        fallback_seed : int
            Random seed for the random_sample fallback strategy.
        """
        selected_mask = scores >= threshold
        selected_by_fallback = False
        if selected_mask.sum() == 0:
            selected_by_fallback = True
            k = max(1, min(fallback_k, len(scores)))
            if fallback_strategy == "random_sample":
                rng = np.random.default_rng(fallback_seed)
                random_indices = rng.choice(len(scores), size=k, replace=False)
                selected_mask = np.zeros_like(scores, dtype=bool)
                selected_mask[random_indices] = True
            else:  # top_k (default)
                top_indices = np.argsort(scores)[-k:]
                selected_mask = np.zeros_like(scores, dtype=bool)
                selected_mask[top_indices] = True

        selected_features = features[selected_mask]
        distances = self.score_samples(selected_features)
        cutoff = float(self.cutoff_distance or 0.0)
        alpha_hat = float((distances > cutoff).mean()) if len(distances) else 0.0
        return NoveltySummary(
            alpha_hat=alpha_hat,
            selected_count=int(selected_mask.sum()),
            selected_by_fallback=selected_by_fallback,
            cutoff_distance=cutoff,
        )


def proxy_a_distance(source_features: np.ndarray, target_features: np.ndarray, seed: int) -> float:
    features = np.vstack([source_features, target_features])
    labels = np.concatenate(
        [
            np.zeros(source_features.shape[0], dtype=int),
            np.ones(target_features.shape[0], dtype=int),
        ]
    )
    if len(np.unique(labels)) < 2:
        return 0.0
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    model = LogisticRegression(
        solver="liblinear",
        max_iter=200,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"sklearn\.utils\.extmath",
        )
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
    err = 1.0 - accuracy_score(test_y, predicted)
    return float(max(0.0, 2.0 * (1.0 - 2.0 * err)))


def average_wasserstein_1d(source_features: np.ndarray, target_features: np.ndarray) -> float:
    if source_features.size == 0 or target_features.size == 0:
        return 0.0
    n_quantiles = max(8, min(128, min(len(source_features), len(target_features))))
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    distances = []
    for column in range(source_features.shape[1]):
        source_q = np.quantile(source_features[:, column], quantiles)
        target_q = np.quantile(target_features[:, column], quantiles)
        distances.append(np.abs(source_q - target_q).mean())
    return float(np.mean(distances))


def standardized_matrices(
    source_train: pd.DataFrame,
    source_eval: pd.DataFrame,
    target_eval: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sanitizer = SourceFeatureSanitizer(feature_columns).fit(source_train[feature_columns])
    sanitized_train = sanitizer.transform(source_train[feature_columns])
    sanitized_source = sanitizer.transform(source_eval[feature_columns])
    sanitized_target = sanitizer.transform(target_eval[feature_columns])
    scaler = BoundedFeatureScaler(feature_columns).fit(sanitized_train)
    train_x = scaler.transform(sanitized_train).to_numpy()
    source_x = scaler.transform(sanitized_source).to_numpy()
    target_x = scaler.transform(sanitized_target).to_numpy()
    return train_x, source_x, target_x
