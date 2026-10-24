
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class FittedPCABaseline:
    

    mean: np.ndarray
    components: np.ndarray
    fit_sample_ids: tuple[str, ...]
    requested_components: int

    @property
    def actual_components(self) -> int:
        return int(self.components.shape[0])

    def reconstruct(self, signal: np.ndarray) -> np.ndarray:
        centered = np.asarray(signal, dtype=np.float64) - self.mean
        scores = self.components @ centered
        return self.mean + self.components.T @ scores


def downsample_interpolate(signal: np.ndarray, ratio: float) -> np.ndarray:
    
    _validate_ratio(ratio)
    vector = np.asarray(signal, dtype=np.float64)
    retained = max(2, min(len(vector), int(round(len(vector) * ratio))))
    indices = np.linspace(0, len(vector) - 1, retained, dtype=int)
    return np.interp(np.arange(len(vector)), indices, vector[indices])


_RANDOM_PROJECTION_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def random_projection_reconstruct(signal: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    
    _validate_ratio(ratio)
    vector = np.asarray(signal, dtype=np.float64)
    retained = max(1, min(len(vector), int(round(len(vector) * ratio))))
    key = (len(vector), retained, int(seed))
    operator = _RANDOM_PROJECTION_CACHE.get(key)
    if operator is None:
        rng = np.random.default_rng(seed)
        projection = rng.choice([-1.0, 1.0], size=(retained, len(vector))) / np.sqrt(retained)
        operator = np.linalg.pinv(projection) @ projection
        _RANDOM_PROJECTION_CACHE[key] = operator
    return operator @ vector


def fit_pca_baseline(
    signals: list[np.ndarray],
    sample_ids: list[str],
    ratio: float,
    seed: int,
    max_fit_samples: int = 4096,
) -> FittedPCABaseline:
    




    _validate_ratio(ratio)
    if len(signals) != len(sample_ids) or not signals:
        raise ValueError("signals and sample_ids must be non-empty and have equal length")
    if max_fit_samples < 1:
        raise ValueError("max_fit_samples must be positive")
    signal_length = len(signals[0])
    if any(len(signal) != signal_length for signal in signals):
        raise ValueError("all PCA training signals must have equal length")
    rng = np.random.default_rng(seed)
    selected = np.arange(len(signals))
    if len(selected) > max_fit_samples:
        selected = np.sort(rng.choice(selected, size=max_fit_samples, replace=False))
    matrix = np.stack([np.asarray(signals[index], dtype=np.float64) for index in selected])
    requested_components = max(1, min(signal_length, int(round(signal_length * ratio))))
    actual_components = min(requested_components, matrix.shape[0], matrix.shape[1])
    solver = "randomized" if actual_components < min(matrix.shape) else "full"
    estimator = PCA(n_components=actual_components, svd_solver=solver, random_state=seed)
    estimator.fit(matrix)
    return FittedPCABaseline(
        mean=np.asarray(estimator.mean_, dtype=np.float64),
        components=np.asarray(estimator.components_, dtype=np.float64),
        fit_sample_ids=tuple(sample_ids[index] for index in selected),
        requested_components=requested_components,
    )


def save_pca_baseline(path: Path, model: FittedPCABaseline) -> None:
    
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            mean=model.mean,
            components=model.components,
            fit_sample_ids=np.asarray(model.fit_sample_ids),
            requested_components=np.asarray(model.requested_components),
        )
    temporary.replace(path)


def load_pca_baseline(path: Path) -> FittedPCABaseline:
    
    with np.load(path, allow_pickle=False) as payload:
        return FittedPCABaseline(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            components=np.asarray(payload["components"], dtype=np.float64),
            fit_sample_ids=tuple(str(value) for value in payload["fit_sample_ids"].tolist()),
            requested_components=int(payload["requested_components"]),
        )


def _validate_ratio(ratio: float) -> None:
    if not 0 < ratio <= 1:
        raise ValueError("compression ratio must be in (0, 1]")
