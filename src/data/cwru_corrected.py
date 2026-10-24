from __future__ import annotations

import hashlib
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from src.data.cwru_loader import SignalSample, infer_label_from_path, normalize_signal, sliding_window
from src.utils.reproducibility import file_sha256

VERSION = "cwru_explicit_de_resampled_v2"
NORMAL_KEYS = {"1797": "X097_DE_time", "1772": "X098_DE_time", "1750": "X099_DE_time", "1730": "X100_DE_time"}
RATE_SOURCE = "Smith and Randall 2015, Table A1, doi:10.1016/j.ymssp.2015.04.021"


def signal_sha256(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype="<f8").tobytes()).hexdigest()


def read_source(root: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    path = (root / row["relative_path"]).resolve()
    path.relative_to(root.resolve())
    if file_sha256(path) != row["file_sha256"]:
        raise ValueError(f"Source changed: {path}")
    raw = np.asarray(loadmat(path)[row["signal_key"]], dtype=np.float64).reshape(-1)
    if not np.isfinite(raw).all() or signal_sha256(raw) != row["raw_signal_sha256"]:
        raise ValueError(f"Selected channel changed: {path}")
    original, target = row["native_rate_hz"], row["target_rate_hz"]
    divisor = gcd(original, target)
    converted = resample_poly(raw, target // divisor, original // divisor) if original != target else raw
    if len(converted) != row["processed_samples"] or signal_sha256(converted) != row["processed_signal_sha256"]:
        raise ValueError(f"Resampling differs from locked input: {path}")
    return raw, converted


def build_inventory(root: Path, allowed_path_keywords: list[str], target_rate: int = 12000) -> dict:
    rows = []
    for path in sorted(root.rglob("*.mat")):
        relative = path.relative_to(root).as_posix()
        if not any(term.lower() in relative.lower() for term in allowed_path_keywords):
            continue
        label = infer_label_from_path(path)
        if label is None:
            continue
        mat = loadmat(path)
        candidates = [k for k in mat if k.endswith("_DE_time")]
        if "NormalBaseline" in path.parts:
            key = NORMAL_KEYS[path.parent.name]
            native = 48000
        elif "12DriveEndFault" in path.parts:
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous fault channel, explicit mapping required: {path}: {candidates}")
            key, native = candidates[0], 12000
        else:
            raise ValueError(f"Sampling rate requires explicit source mapping: {path}")
        raw = np.asarray(mat[key], dtype=np.float64).reshape(-1)
        divisor = gcd(native, target_rate)
        converted = resample_poly(raw, target_rate // divisor, native // divisor) if native != target_rate else raw
        if label not in {"normal", "ball", "inner_race", "outer_race"} or not np.isfinite(raw).all():
            raise ValueError(f"Invalid source: {path}")
        rows.append({"relative_path": relative, "signal_key": key, "available_de_keys": candidates,
                     "label": label, "native_rate_hz": native, "target_rate_hz": target_rate,
                     "native_rate_source": RATE_SOURCE if native == 48000 else "CWRU 12DriveEndFault collection",
                     "file_sha256": file_sha256(path), "raw_signal_sha256": signal_sha256(raw),
                     "raw_samples": len(raw), "processed_samples": len(converted),
                     "processed_signal_sha256": signal_sha256(converted)})
    if not rows:
        raise ValueError("No CWRU sources selected")
    return {"version": VERSION, "resampling": "scipy.signal.resample_poly; window=('kaiser',5.0); padtype=constant",
            "sources": rows}


def load_corrected_segments(root: Path, inventory: dict, segment_length: int = 1024,
                            stride: int = 1024, normalize: str = "max_abs") -> list[SignalSample]:
    if inventory["version"] != VERSION:
        raise ValueError("Unknown CWRU preprocessing version")
    seen_raw, seen_windows, samples = {}, {}, []
    for row in inventory["sources"]:
        relative = row["relative_path"]
        digest = row["raw_signal_sha256"]
        if digest in seen_raw:
            raise ValueError(f"Duplicate selected channels: {seen_raw[digest]} and {relative}")
        seen_raw[digest] = relative
        _, vector = read_source(root, row)
        source = str((root / relative).resolve())
        windows = sliding_window(vector, segment_length, stride)
        if not windows:
            raise ValueError(f"Source has no complete windows: {relative}")
        for index, window in enumerate(windows):
            normalized = normalize_signal(window, normalize)
            digest = signal_sha256(normalized)
            previous = seen_windows.get(digest)
            if previous is not None and previous[0] != source:
                raise ValueError(f"Duplicate normalized windows: {previous} and {(source, index)}")
            seen_windows[digest] = (source, index)
            samples.append(SignalSample(signal=normalized, label=row["label"], source_file=source,
                                        sampling_rate=row["target_rate_hz"], sensor=row["signal_key"],
                                        segment_id=index, group_id=source))
    return samples
