
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.fftpack import dct

from src.utils.io_tools import export_json
from src.utils.reproducibility import canonical_hash, file_sha256


PRIOR_VERSION = 1
PRIOR_FORMULA = "class-balanced Fisher on log1p(abs(orthonormal DCT) / signal RMS)"


def select_training_prior_indices(
    labels: Sequence[str],
    split_map: dict[int, str],
    sample_ids: Sequence[str],
    seed: int,
    max_per_class: int,
) -> list[int]:
    
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive")
    by_class: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        if split_map[index] == "train":
            by_class.setdefault(str(label), []).append(index)
    if not by_class:
        raise ValueError("cannot fit a diagnostic prior without training samples")
    selected: list[int] = []
    for label, indices in sorted(by_class.items()):
        ranked = sorted(
            indices,
            key=lambda index: canonical_hash(
                {"seed": int(seed), "label": label, "sample_id": sample_ids[index]}, length=32
            ),
        )
        selected.extend(ranked[:max_per_class])
    return sorted(selected, key=lambda index: (str(labels[index]), str(sample_ids[index])))


def fisher_atom_weights(
    class_sums: np.ndarray,
    class_square_sums: np.ndarray,
    class_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    
    sums = np.asarray(class_sums, dtype=np.float64)
    square_sums = np.asarray(class_square_sums, dtype=np.float64)
    counts = np.asarray(class_counts, dtype=np.int64)
    if sums.ndim != 2 or square_sums.shape != sums.shape:
        raise ValueError("class moment arrays must have matching [class, atom] shapes")
    if counts.shape != (sums.shape[0],) or np.any(counts < 2):
        raise ValueError("every class needs at least two training samples for a Fisher prior")
    means = sums / counts[:, None]
    variances = np.maximum(square_sums / counts[:, None] - means**2, 0.0)
    grand_mean = means.mean(axis=0)
    between = np.mean((means - grand_mean) ** 2, axis=0)
    within = variances.mean(axis=0)
    raw = between / (within + 1e-12)
    finite = raw[np.isfinite(raw)]
    scale = float(np.quantile(finite, 0.99)) if finite.size else 0.0
    weights = np.zeros_like(raw) if scale <= 0 else np.clip(raw / scale, 0.0, 1.0)
    return weights.astype(np.float64), raw.astype(np.float64)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _validate_cached_prior(root: Path, expected_signature: str) -> tuple[np.ndarray, dict] | None:
    array_path = root / "diagnostic_prior.npz"
    metadata_path = root / "diagnostic_prior.json"
    if not array_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("complete") or metadata.get("prior_signature") != expected_signature:
        raise RuntimeError(f"incompatible diagnostic-prior cache: {metadata_path}")
    if metadata.get("npz_sha256") != file_sha256(array_path):
        raise RuntimeError(f"diagnostic-prior hash mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as payload:
        weights = np.asarray(payload["atom_weights"], dtype=np.float64)
        stored_signature = str(payload["prior_signature"].item())
    if stored_signature != expected_signature or weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise RuntimeError(f"invalid diagnostic-prior payload: {array_path}")
    return weights, metadata


def fit_or_load_diagnostic_prior(
    root: Path,
    samples: Sequence,
    split_map: dict[int, str],
    sample_ids: Sequence[str],
    seed: int,
    max_per_class: int = 4096,
    checkpoint_every: int = 256,
) -> tuple[np.ndarray, dict]:
    
    labels = [str(sample.label) for sample in samples]
    selected = select_training_prior_indices(labels, split_map, sample_ids, seed, max_per_class)
    class_names = sorted({labels[index] for index in selected})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    selected_class_counts = {
        name: sum(labels[index] == name for index in selected) for name in class_names
    }
    if any(count < 2 for count in selected_class_counts.values()):
        raise ValueError(
            "every diagnostic-prior class needs at least two selected training samples: "
            f"{selected_class_counts}"
        )
    selected_ids = [str(sample_ids[index]) for index in selected]
    signal_length = int(np.asarray(samples[selected[0]].signal).size)
    config = {
        "version": PRIOR_VERSION,
        "formula": PRIOR_FORMULA,
        "fit_scope": "training_only",
        "seed": int(seed),
        "max_samples_per_class": int(max_per_class),
        "class_names": class_names,
        "sample_count": len(selected),
        "selected_class_counts": selected_class_counts,
        "selected_sample_ids_hash": canonical_hash(selected_ids, length=32),
        "signal_length": signal_length,
        "sparse_basis": "dct",
    }
    prior_signature = canonical_hash(config)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cached = _validate_cached_prior(root, prior_signature)
    if cached is not None:
        print(f"diagnostic prior already complete: {root / 'diagnostic_prior.npz'}")
        return cached

    progress_path = root / "diagnostic_prior_progress.npz"
    class_sums = np.zeros((len(class_names), signal_length), dtype=np.float64)
    class_square_sums = np.zeros_like(class_sums)
    class_counts = np.zeros(len(class_names), dtype=np.int64)
    next_position = 0
    if progress_path.exists():
        with np.load(progress_path, allow_pickle=False) as progress:
            if str(progress["prior_signature"].item()) != prior_signature:
                raise RuntimeError(f"incompatible diagnostic-prior progress: {progress_path}")
            class_sums = np.asarray(progress["class_sums"], dtype=np.float64)
            class_square_sums = np.asarray(progress["class_square_sums"], dtype=np.float64)
            class_counts = np.asarray(progress["class_counts"], dtype=np.int64)
            next_position = int(progress["next_position"].item())
        print(f"resuming diagnostic prior at {next_position}/{len(selected)}")

    def save_progress(position: int) -> None:
        _atomic_npz(
            progress_path,
            prior_signature=np.asarray(prior_signature),
            next_position=np.asarray(position, dtype=np.int64),
            class_sums=class_sums,
            class_square_sums=class_square_sums,
            class_counts=class_counts,
        )

    completed = next_position
    try:
        for position in range(next_position, len(selected)):
            index = selected[position]
            signal = np.asarray(samples[index].signal, dtype=np.float64).reshape(-1)
            if signal.size != signal_length:
                raise ValueError("diagnostic-prior signals must share one fixed length")
            rms = float(np.sqrt(np.mean(signal**2)) + 1e-12)
            features = np.log1p(np.abs(dct(signal, norm="ortho")) / rms)
            class_index = class_to_index[labels[index]]
            class_sums[class_index] += features
            class_square_sums[class_index] += features**2
            class_counts[class_index] += 1
            completed = position + 1
            if completed % checkpoint_every == 0 or completed == len(selected):
                save_progress(completed)
                print(f"[diagnostic prior] {completed}/{len(selected)} samples")
    except BaseException:
        save_progress(completed)
        raise

    weights, raw_scores = fisher_atom_weights(class_sums, class_square_sums, class_counts)
    array_path = root / "diagnostic_prior.npz"
    _atomic_npz(
        array_path,
        prior_signature=np.asarray(prior_signature),
        atom_weights=weights,
        raw_fisher_scores=raw_scores,
        class_names=np.asarray(class_names),
        class_counts=class_counts,
        selected_sample_ids=np.asarray(selected_ids),
    )
    metadata = {
        "complete": True,
        "prior_signature": prior_signature,
        "config": config,
        "class_counts": {name: int(class_counts[index]) for index, name in enumerate(class_names)},
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "npz_file": array_path.name,
        "npz_sha256": file_sha256(array_path),
    }
    export_json(root / "diagnostic_prior.json", metadata)
    if progress_path.exists():
        progress_path.unlink()
    return weights, metadata
