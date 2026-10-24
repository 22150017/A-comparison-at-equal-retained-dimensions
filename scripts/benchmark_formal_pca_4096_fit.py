




from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cwru_loader import normalize_signal
from src.data.pu_loader import PU_SAMPLING_RATE, extract_vibration, resample_signal
from src.reconstruction.compression_baselines import fit_pca_baseline
from src.utils.reproducibility import canonical_hash, file_sha256


FORMAL_ROOT = (
    ROOT / "artifacts_v2" / "datasets" / "pu"
    / "baseline_pca_stft_r0.50_s42" / "a7e5aecf3ad8b57b"
)
DEFAULT_OUTPUT = Path(r"F:\审查\outputs\pca_4096_fit_timing_pu_seed42.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    manifest_path = args.formal_root / "dataset_manifest.json"
    metadata_path = args.formal_root / "pca_training_only.json"
    fitted_path = args.formal_root / "pca_training_only.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    condition = manifest["condition"]
    signal = condition["signal"]
    if (condition["dataset"], condition["method"], condition["seed"], condition["compression_ratio"]) != ("pu", "baseline:pca", 42, 0.5):
        raise ValueError("This is not the locked formal PU seed-42 PCA condition")
    if not manifest["complete"] or metadata["fit_scope"] != "training_only":
        raise ValueError("Formal PCA manifest is incomplete or not training-only")
    pca_lock = condition["phase2"]["pca"]
    if pca_lock["fit_sample_count"] != 4096 or pca_lock["max_fit_samples"] != 4096:
        raise ValueError("Locked PCA fitting count or cap is not 4096")
    if metadata["requested_components"] != 512 or metadata["actual_components"] != 512:
        raise ValueError("Formal PCA metadata does not specify 512 components")
    formal_model_sha256 = file_sha256(fitted_path)
    if formal_model_sha256 != metadata["sha256"]:
        raise ValueError("Formal PCA model SHA256 does not match pca_training_only.json")

    sample_ids = tuple(metadata["fit_sample_ids"])
    if len(sample_ids) != 4096 or len(set(sample_ids)) != 4096:
        raise ValueError("Expected exactly 4096 distinct formal fit-window IDs")
    if canonical_hash(sample_ids) != pca_lock["fit_sample_ids_hash"]:
        raise ValueError("Formal fit-window IDs do not match the locked manifest")
    with np.load(fitted_path, allow_pickle=False) as formal:
        fitted_ids = tuple(str(value) for value in formal["fit_sample_ids"].tolist())
        formal_mean = np.asarray(formal["mean"], dtype=np.float64)
        formal_components = int(formal["components"].shape[0])
    if fitted_ids != sample_ids or formal_components != 512:
        raise ValueError("Formal fitted PCA does not match its metadata")

    selected = set(sample_ids)
    records = {record["sample_id"]: record for record in manifest["records"] if record["sample_id"] in selected}
    if len(records) != 4096 or any(record["split"] != "train" for record in records.values()):
        raise ValueError("A formal PCA fit window is missing or outside the training split")
    grouped: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for position, sample_id in enumerate(sample_ids):
        record = records[sample_id]
        grouped[Path(record["source_file"])].append((position, int(record["segment_id"])))
    formal_source_hashes = {
        str(Path(item["path"]).resolve()): item["sha256"]
        for item in condition["sources"]
    }
    if any(str(path.resolve()) not in formal_source_hashes for path in grouped):
        raise ValueError("A selected source file is absent from the formal manifest")

    preview = {
        "formal_manifest": str(manifest_path),
        "formal_pca_metadata": str(metadata_path),
        "formal_pca_model": str(fitted_path),
        "dataset": "PU",
        "seed": 42,
        "training_windows": len(sample_ids),
        "training_source_files": len(grouped),
        "components": formal_components,
        "repeats": args.repeats,
        "output": str(args.output),
        "timed_scope": "fit_pca_baseline only; raw reading and preprocessing excluded",
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    window_length = int(signal["segment_length"])
    stride = int(signal["stride"])
    target_rate = int(signal["sampling_rate"])
    matrix = np.empty((len(sample_ids), window_length), dtype=np.float64)
    preprocessing_started = time.perf_counter()
    for number, (path, positions) in enumerate(grouped.items(), 1):
        if file_sha256(path) != formal_source_hashes[str(path.resolve())]:
            raise ValueError(f"Raw source SHA256 mismatch: {path}")
        vector = resample_signal(extract_vibration(path), PU_SAMPLING_RATE, target_rate)
        for row, segment_id in positions:
            start = segment_id * stride
            window = vector[start : start + window_length]
            if len(window) != window_length:
                raise ValueError(f"Window {segment_id} is out of range: {path}")
            matrix[row] = normalize_signal(window, signal["normalize"])
        if number % 100 == 0 or number == len(grouped):
            print(f"[materialize] {number}/{len(grouped)} training source files")
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    input_mean_difference = float(np.max(np.abs(matrix.mean(axis=0) - formal_mean)))
    if input_mean_difference > 1e-9:
        raise ValueError(f"Materialized windows differ from the formal PCA fit: {input_mean_difference}")

    signals = list(matrix)
    elapsed: list[float] = []
    for repetition in range(args.repeats):
        started = time.perf_counter()
        fitted = fit_pca_baseline(signals, list(sample_ids), 0.5, 42, 4096)
        seconds = time.perf_counter() - started
        if fitted.fit_sample_ids != sample_ids or fitted.actual_components != 512:
            raise ValueError("Timed PCA fit used an unexpected sample set or component count")
        elapsed.append(seconds)
        print(f"[PCA fit {repetition + 1}/{args.repeats}] {seconds:.4f} s")

    output = {
        **preview,
        "manifest_sha256": file_sha256(manifest_path),
        "pca_metadata_sha256": file_sha256(metadata_path),
        "formal_pca_model_sha256": formal_model_sha256,
        "verified_raw_source_file_count": len(grouped),
        "fit_sample_ids_hash": canonical_hash(sample_ids),
        "input_mean_max_abs_difference_vs_formal": input_mean_difference,
        "fit_seconds_each_run": elapsed,
        "fit_seconds_first_run": elapsed[0],
        "fit_seconds_median": float(np.median(elapsed)),
        "fit_seconds_mean": float(np.mean(elapsed)),
        "preprocessing_seconds_excluded_from_fit": preprocessing_seconds,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "pca_fit_performed": True,
        "classifier_training_performed": False,
        "classifier_inference_performed": False,
        "independent_test_signal_values_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"fit_seconds_median": output["fit_seconds_median"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
