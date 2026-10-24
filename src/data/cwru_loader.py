from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.io import loadmat


CWRU_KEY_HINTS = (
    "DE_time",
    "FE_time",
    "BA_time",
)


@dataclass
class SignalSample:
    signal: np.ndarray
    label: str
    source_file: str
    load_hp: int | None = None
    sampling_rate: int | None = None
    sensor: str | None = None
    segment_id: int | None = None
    
    group_id: str | None = None


def sliding_window(signal: np.ndarray, window: int, stride: int) -> List[np.ndarray]:
    if window <= 0 or stride <= 0 or not isinstance(window, int) or not isinstance(stride, int):
        raise ValueError("window and stride must be positive integers")
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError("signal must be a finite one-dimensional vector")
    windows: List[np.ndarray] = []
    if signal.shape[0] < window:
        return windows
    for start in range(0, signal.shape[0] - window + 1, stride):
        windows.append(signal[start : start + window].copy())
    return windows


def normalize_signal(signal: np.ndarray, mode: str = "max_abs") -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("normalization requires a nonempty finite one-dimensional vector")
    if mode == "max_abs":
        denom = np.max(np.abs(signal)) + 1e-12
        return signal / denom
    if mode == "zscore":
        return (signal - signal.mean()) / (signal.std() + 1e-12)
    if mode not in {"none", "identity"}:
        raise ValueError(f"unknown normalization mode: {mode}")
    return signal.astype(np.float64)


def infer_label_from_text(text: str) -> str | None:
    lower = text.lower()
    tokens = re.split(r"[^a-z0-9]+", lower)
    tokens = [token.rstrip("0123456789") for token in tokens]
    if any(tok in {"normal", "baseline", "healthy", "normalbaseline"} for tok in tokens):
        return "normal"
    if any(tok in {"ball", "ballfault", "ballfaults"} for tok in tokens) or re.search(r"(^|[^a-z])b(\d|$)", lower):
        return "ball"
    if any(tok in {"inner", "innerrace", "innerracefault", "innerracefaults"} for tok in tokens) or re.search(r"(^|[^a-z])ir(\d|$|[^a-z0-9])", lower):
        return "inner_race"
    if any(tok in {"outer", "outerrace", "outerracefault", "outerracefaults"} for tok in tokens) or re.search(r"(^|[^a-z])or(\d|$|[^a-z0-9])", lower):
        return "outer_race"
    return None


def infer_label_from_path(path: Path) -> str | None:
    parts = [path.stem, *reversed(path.parent.parts)]
    for part in parts:
        label = infer_label_from_text(part)
        if label is not None:
            return label
    return None


def infer_load_hp_from_name(name: str) -> int | None:
    lower = name.lower()
    for hp in (0, 1, 2, 3):
        if f"_{hp}" in lower or f"load{hp}" in lower or f"{hp}hp" in lower:
            return hp
    return None


def infer_sampling_rate(root: Path, file_name: str, default: int = 12000) -> int:
    joined = f"{root} {file_name}".lower()
    if "48k" in joined or "48000" in joined:
        return 48000
    if "12k" in joined or "12000" in joined:
        return 12000
    return default


def choose_signal_key(mat_dict: Dict, prefer_sensor: str = "DE") -> str | None:
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]

    sensor_order = [prefer_sensor.upper(), "DE", "FE", "BA"]
    for sensor in sensor_order:
        for key in keys:
            if sensor in key.upper() and key.endswith("_time"):
                return key

    for hint in CWRU_KEY_HINTS:
        for key in keys:
            if hint.lower() in key.lower():
                return key

    for key in keys:
        value = mat_dict[key]
        if isinstance(value, np.ndarray) and value.ndim in (1, 2):
            flat = value.reshape(-1)
            if flat.size > 1024:
                return key
    return None


def load_cwru_mat_file(path: Path, prefer_sensor: str = "DE") -> tuple[np.ndarray, str]:
    mat = loadmat(path)
    key = choose_signal_key(mat, prefer_sensor=prefer_sensor)
    if key is None:
        raise ValueError(f"Could not extract a usable vibration vector from {path}")
    vector = mat[key].reshape(-1).astype(np.float64)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"nonfinite or empty vibration vector: {path}")
    return vector, key


def load_cwru_segments(
    root: Path,
    segment_length: int = 1024,
    stride: int = 1024,
    normalize: str = "max_abs",
    prefer_sensor: str = "DE",
    allowed_classes: list[str] | None = None,
    allowed_sampling_rate: int | None = 12000,
    allowed_path_keywords: list[str] | None = None,
) -> List[SignalSample]:
    samples: List[SignalSample] = []
    for mat_path in sorted(root.rglob("*.mat")):
        if allowed_path_keywords:
            joined_path = str(mat_path).lower()
            if not any(keyword.lower() in joined_path for keyword in allowed_path_keywords):
                continue
        label = infer_label_from_path(mat_path)
        if label is None:
            warnings.warn(f"unrecognized CWRU label; file excluded: {mat_path}", RuntimeWarning)
            continue
        if allowed_classes and label not in allowed_classes:
            continue

        sampling_rate = infer_sampling_rate(mat_path.parent, mat_path.name)
        if allowed_sampling_rate is not None and sampling_rate != allowed_sampling_rate:
            continue

        signal, key = load_cwru_mat_file(mat_path, prefer_sensor=prefer_sensor)
        load_hp = infer_load_hp_from_name(mat_path.stem)
        windows = sliding_window(signal, segment_length, stride)
        for idx, window in enumerate(windows):
            samples.append(
                SignalSample(
                    signal=normalize_signal(window, normalize),
                    label=label,
                    source_file=str(mat_path),
                    load_hp=load_hp,
                    sampling_rate=sampling_rate,
                    sensor=key,
                    segment_id=idx,
                    group_id=str(mat_path.resolve()),
                )
            )
    return samples


def grouped_counts(samples: Iterable[SignalSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    return counts


def dataset_summary(samples: Iterable[SignalSample]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        label_group = summary.setdefault(sample.label, {})
        rate_key = str(sample.sampling_rate or "unknown")
        label_group[rate_key] = label_group.get(rate_key, 0) + 1
    return summary
