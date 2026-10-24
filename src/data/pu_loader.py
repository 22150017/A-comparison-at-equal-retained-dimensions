












from __future__ import annotations

import re
import warnings
from math import gcd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from src.data.cwru_loader import SignalSample, normalize_signal, sliding_window



LEGACY_BEARING_LABEL_MAP = {
    
    "K001": "normal", "K002": "normal", "K003": "normal",
    "K004": "normal", "K005": "normal", "K006": "normal",
    
    "KA01": "outer_race", "KA03": "outer_race", "KA05": "outer_race",
    "KA06": "outer_race", "KA07": "outer_race", "KA08": "outer_race",
    "KA09": "outer_race",
    
    "KI01": "inner_race", "KI03": "inner_race", "KI05": "inner_race",
    "KI07": "inner_race", "KI08": "inner_race",
    
    "KA04": "outer_race", "KA15": "outer_race", "KA16": "outer_race",
    "KA22": "outer_race", "KA30": "outer_race",
    
    "KB23": "ball", "KB24": "ball", "KB27": "ball",
    
    "KI04": "inner_race", "KI14": "inner_race", "KI16": "inner_race",
    "KI17": "inner_race", "KI18": "inner_race", "KI21": "inner_race",
}



BEARING_LABEL_MAP = {
    **LEGACY_BEARING_LABEL_MAP,
    "KB23": "compound", "KB24": "compound", "KB27": "compound",
}
PU_LABEL_SCHEMA = "official_v2"


def bearing_label_map(label_schema: str = PU_LABEL_SCHEMA) -> dict[str, str]:
    if label_schema == "official_v2":
        return dict(BEARING_LABEL_MAP)
    if label_schema == "legacy_v1":
        warnings.warn(
            "legacy_v1 reproduces the historical INCORRECT KB->ball label. "
            "Use only for historical artifact reproduction, not new scientific claims.",
            UserWarning, stacklevel=2,
        )
        return dict(LEGACY_BEARING_LABEL_MAP)
    raise ValueError(f"unknown PU label schema: {label_schema}")


OPERATING_CONDITIONS = {
    "N15_M07_F10": (1500, 0.7, 1000),
    "N15_M07_F04": (1500, 0.7, 400),
    "N15_M01_F10": (1500, 0.1, 1000),
    "N09_M07_F10": (900,  0.7, 1000),
}

PU_SAMPLING_RATE = 64000  
CWRU_SAMPLING_RATE = 12000  


def extract_vibration(mat_path: Path) -> np.ndarray:
    
    mat = loadmat(str(mat_path))

    
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No data key found in {mat_path}")

    data = mat[keys[0]]
    Y = data["Y"][0, 0]
    Y_data = Y["Data"]  

    
    vibration = Y_data[0, 6].flatten().astype(np.float64)
    if vibration.size == 0 or not np.all(np.isfinite(vibration)):
        raise ValueError(f"nonfinite or empty vibration vector: {mat_path}")
    return vibration


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    
    if orig_sr <= 0 or target_sr <= 0 or not np.all(np.isfinite(signal)):
        raise ValueError("sampling rates must be positive and signal finite")
    if orig_sr == target_sr:
        return signal
    divisor = gcd(orig_sr, target_sr)
    return resample_poly(signal, target_sr // divisor, orig_sr // divisor)


def parse_filename(filename: str) -> dict:
    
    
    pattern = r"(N\d+)_(M\d+)_(F\d+)_(K\w+?)_(\d+)\.mat"
    match = re.match(pattern, filename)
    if not match:
        return {}
    return {
        "operating_condition": f"{match.group(1)}_{match.group(2)}_{match.group(3)}",
        "bearing_code": match.group(4),
        "measurement_id": int(match.group(5)),
    }


def load_pu_segments(
    root: Path,
    segment_length: int = 1024,
    stride: int = 1024,
    normalize: str = "max_abs",
    allowed_classes: list[str] | None = None,
    target_sampling_rate: int = CWRU_SAMPLING_RATE,
    categories: list[str] | None = None,
    label_schema: str = PU_LABEL_SCHEMA,
) -> List[SignalSample]:
    


















    label_map = bearing_label_map(label_schema)
    if allowed_classes:
        unknown = set(allowed_classes) - set(label_map.values())
        if unknown:
            raise ValueError(
                f"Classes {sorted(unknown)} are not in PU {label_schema}; "
                "KB23/24/27 are compound raceway damage, not ball defects. "
                "Use a corrected configuration; legacy_v1 is historical reproduction only."
            )
    samples: List[SignalSample] = []

    if categories is None:
        categories = ["Healthy", "Artificial_damages", "Real_damages"]

    for category in categories:
        cat_dir = root / category
        if not cat_dir.exists():
            continue

        for bearing_dir in sorted(cat_dir.iterdir()):
            if not bearing_dir.is_dir():
                continue

            bearing_code = bearing_dir.name
            label = label_map.get(bearing_code)
            if label is None:
                continue
            if allowed_classes and label not in allowed_classes:
                continue

            for mat_path in sorted(bearing_dir.glob("*.mat")):
                info = parse_filename(mat_path.name)
                if not info:
                    continue

                try:
                    vibration = extract_vibration(mat_path)
                except Exception as e:
                    print(f"  Warning: failed to read {mat_path.name}: {e}")
                    continue

                
                vibration = resample_signal(vibration, PU_SAMPLING_RATE, target_sampling_rate)

                
                windows = sliding_window(vibration, segment_length, stride)

                for idx, window in enumerate(windows):
                    samples.append(
                        SignalSample(
                            signal=normalize_signal(window, normalize),
                            label=label,
                            source_file=str(mat_path),
                            load_hp=None,
                            sampling_rate=target_sampling_rate,
                            sensor="vibration_1",
                            segment_id=idx,
                            group_id=bearing_code,
                        )
                    )

    return samples
