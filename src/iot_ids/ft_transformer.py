from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from iot_ids.modeling import BoundedFeatureScaler, SourceFeatureSanitizer

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def torch_is_available() -> bool:
    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "ft_transformer requires torch. Install the optional ft extra, for example "
            "`pip install -e '.[ft]'`."
        )


def _hash_key(value: str, seed: int) -> tuple[int, str]:
    payload = f"{seed}:{value}".encode()
    digest = int.from_bytes(payload, "little", signed=False)
    return digest, value


def _split_groups(
    groups: pd.Series,
    seed: int,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    group_values = sorted(groups.astype(str).unique(), key=lambda value: _hash_key(value, seed))
    if len(group_values) < 2:
        all_idx = np.arange(len(groups))
        return all_idx, all_idx

    n_val_groups = max(1, int(round(len(group_values) * val_frac)))
    if n_val_groups >= len(group_values):
        n_val_groups = len(group_values) - 1
    val_groups = set(group_values[:n_val_groups])
    train_mask = ~groups.astype(str).isin(val_groups)
    val_mask = groups.astype(str).isin(val_groups)
    train_idx = np.flatnonzero(train_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    if len(train_idx) == 0 or len(val_idx) == 0:
        all_idx = np.arange(len(groups))
        midpoint = max(1, len(all_idx) // 5)
        return all_idx[midpoint:], all_idx[:midpoint]
    return train_idx, val_idx


@dataclass(frozen=True)
class FTTransformerConfig:
    d_token: int = 32
    n_heads: int = 4
    n_blocks: int = 2
    ff_hidden: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 8
    patience: int = 2
    inner_val_frac: float = 0.20
    max_train_batches_per_epoch: int = 64
    max_train_rows: int = 50_000
    inference_batch_size: int = 8_192


def _cap_training_rows(
    frame: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    seed: int,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if len(frame) <= max_rows:
        return frame, labels, groups

    sample_parts: list[pd.DataFrame] = []
    working = frame.assign(_label=labels)
    frac = max_rows / len(working)
    for label_value, subset in working.groupby("_label", sort=True):
        n_rows = min(len(subset), max(1, int(round(len(subset) * frac))))
        sample_parts.append(subset.sample(n=n_rows, random_state=seed + int(label_value)))
    sampled = pd.concat(sample_parts, ignore_index=False)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    sampled = sampled.sort_index()
    sampled_labels = labels.loc[sampled.index].copy()
    sampled_groups = groups.loc[sampled.index].copy()
    return sampled.drop(columns=["_label"]), sampled_labels, sampled_groups


if torch is not None:

    class _NumericTokenizer(nn.Module):
        def __init__(self, n_features: int, d_token: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.empty(n_features, d_token))
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


    class _FTTransformerModule(nn.Module):
        def __init__(self, n_features: int, config: FTTransformerConfig) -> None:
            super().__init__()
            self.tokenizer = _NumericTokenizer(n_features=n_features, d_token=config.d_token)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_token))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_token,
                nhead=config.n_heads,
                dim_feedforward=config.ff_hidden,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_blocks)
            self.norm = nn.LayerNorm(config.d_token)
            self.head = nn.Sequential(
                nn.Linear(config.d_token, config.ff_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ff_hidden, 1),
            )
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            hidden = self.encoder(torch.cat([cls, tokens], dim=1))
            pooled = self.norm(hidden[:, 0, :])
            return self.head(pooled).squeeze(-1)

else:  # pragma: no cover - import-time compatibility when torch is unavailable

    class _FTTransformerModule:  # type: ignore[no-redef]
        def __init__(self, n_features: int, config: FTTransformerConfig) -> None:
            _require_torch()


class FTTransformerEstimator:
    def __init__(
        self,
        feature_columns: Iterable[str],
        seed: int,
        config: FTTransformerConfig | None = None,
    ) -> None:
        _require_torch()
        self.feature_columns = list(feature_columns)
        self.seed = int(seed)
        self.config = config or FTTransformerConfig()
        self.sanitizer = SourceFeatureSanitizer(self.feature_columns)
        self.scaler = BoundedFeatureScaler(self.feature_columns)
        self.device_ = torch.device("cpu")

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
    ) -> FTTransformerEstimator:
        _require_torch()
        self._set_determinism()
        frame = x.loc[:, self.feature_columns].copy()
        labels = pd.Series(y, index=frame.index).astype(float)
        group_series = (
            pd.Series(groups, index=frame.index).astype(str)
            if groups is not None
            else pd.Series(np.arange(len(frame)).astype(str), index=frame.index)
        )
        frame, labels, group_series = _cap_training_rows(
            frame,
            labels,
            group_series,
            seed=self.seed,
            max_rows=self.config.max_train_rows,
        )

        sanitized = self.sanitizer.fit(frame).transform(frame)
        transformed = self.scaler.fit(sanitized).transform(sanitized)
        values = transformed.to_numpy(dtype=np.float32, copy=True)
        labels_np = labels.to_numpy(dtype=np.float32, copy=True)
        self.device_ = torch.device("cpu")

        train_idx, val_idx = _split_groups(
            group_series,
            seed=self.seed,
            val_frac=self.config.inner_val_frac,
        )

        self.model_ = _FTTransformerModule(
            n_features=len(self.feature_columns),
            config=self.config,
        ).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        train_loader = self._make_loader(values[train_idx], labels_np[train_idx], shuffle=True)
        val_features = torch.from_numpy(values[val_idx]).to(self.device_)
        val_labels = torch.from_numpy(labels_np[val_idx]).to(self.device_)

        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        patience_left = self.config.patience

        for _ in range(self.config.max_epochs):
            self.model_.train()
            for batch_index, (batch_x, batch_y) in enumerate(train_loader):
                if batch_index >= self.config.max_train_batches_per_epoch:
                    break
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(val_features)
                val_loss = float(loss_fn(val_logits, val_labels).item())

            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                patience_left = self.config.patience
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        _require_torch()
        frame = self._to_frame(x)
        features = self.scaler.transform(self.sanitizer.transform(frame))
        values = features.to_numpy(dtype=np.float32, copy=True)
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), self.config.inference_batch_size):
                stop = min(len(values), start + self.config.inference_batch_size)
                batch = torch.from_numpy(values[start:stop]).to(self.device_)
                logits = self.model_(batch)
                batches.append(torch.sigmoid(logits).cpu().numpy())
        positive = np.concatenate(batches, axis=0) if batches else np.zeros(0, dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def _make_loader(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, max(1, len(dataset))),
            shuffle=shuffle,
            generator=generator,
        )

    def _set_determinism(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _to_frame(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.loc[:, self.feature_columns].copy()
        return pd.DataFrame(x, columns=self.feature_columns)
