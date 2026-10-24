from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ["no_cs", "ilsomp", "downsample", "pca", "random_projection"]
BENCHMARKS = {
    "pu": ROOT / "artifacts_v2/studies/benchmark/formal_pu_598ba4f57533501db61bcb0c_summary.json",
    "cwru": ROOT / "artifacts_v2/studies/benchmark/external_cwru_five_seed_d3be8e689220c8c902d1e2d3_summary.json",
}
SOURCES = {
    "pu": ROOT / "artifacts_v2/reports/source_file_performance/formal-pu_fe02696922cd_source_file_summary.json",
    "cwru": ROOT / "artifacts_v2/reports/source_file_performance/external-cwru_65b26947512b_source_file_summary.json",
}


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:24]


def artifact(path: str | Path, expected: str | None = None) -> dict:
    path = Path(path).resolve()
    sha = digest(path)
    if expected is not None and sha != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return {"path": str(path), "sha256": sha}


def benchmark_protocol_inputs(dataset: str, benchmark: dict) -> list[dict]:
    if dataset == "pu":
        return [artifact(benchmark["protocol_path"], benchmark["protocol_sha256"])]
    base_summary = artifact(benchmark["base_summary"], benchmark["base_summary_sha256"])
    base = read_json(base_summary["path"])
    return [artifact(benchmark["extension_protocol_path"], benchmark["extension_protocol_sha256"]),
            base_summary, artifact(base["protocol_path"], base["protocol_sha256"])]


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key].copy() for key in payload.files}
    ids, targets, logits, names = (result[key] for key in ("sample_ids", "y_true", "logits", "class_names"))
    if len(set(ids.tolist())) != len(ids) or logits.shape != (len(ids), len(names)):
        raise ValueError(f"Invalid prediction dimensions or duplicate IDs: {path}")
    if targets.shape != (len(ids),) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid targets or nonfinite logits: {path}")
    if not np.issubdtype(targets.dtype, np.integer) or np.any((targets < 0) | (targets >= len(names))):
        raise ValueError(f"Invalid target classes: {path}")
    return result


def align(reference: dict, candidate: dict) -> None:
    for key in ("sample_ids", "y_true", "class_names"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Paired artifact mismatch: {key}")


def ordered_records(manifest: dict, payload: dict) -> list[dict]:
    records = {row["sample_id"]: row for row in manifest["records"] if row["split"] == "test"}
    if set(records) != set(payload["sample_ids"].tolist()):
        raise ValueError("Manifest and prediction test IDs differ")
    result = [records[str(key)] for key in payload["sample_ids"]]
    for row, target in zip(result, payload["y_true"]):
        if row["label"] != str(payload["class_names"][target]):
            raise ValueError("Manifest and prediction targets differ")
    return result


def verify_inputs(protocol: dict) -> None:
    for item in protocol["inputs"]:
        artifact(item["path"], item["sha256"])


def prepare(protocol: dict, dry_run: bool, confirmation: str | None, root: Path) -> Path:
    value = signature(protocol)
    lock_path = root.resolve() / value / "protocol_lock.json"
    preview = {"signature": value, "lock": str(lock_path), "experiment": protocol["experiment"],
               "training_cells": 0, "inference_cells": protocol["inference_cells"],
               "runs": [{"dataset": run["dataset"], "seed": run["seed"], "test_windows": run["test_windows"],
                         "source_files": run["source_files"], "window_lengths": run["window_lengths"]}
                        for run in protocol["runs"]], "uored_test_opened": False}
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not dry_run:
        if confirmation != value:
            raise SystemExit(f"locking requires --confirm-signature {value}")
        lock = {"signature": value, "protocol": protocol}
        if lock_path.exists():
            if read_json(lock_path) != lock:
                raise ValueError("Existing lock differs; no overwrite allowed")
        else:
            save_json(lock_path, lock)
    return lock_path


def load_lock(path: Path, experiment: str) -> dict:
    lock = read_json(path)
    if lock["signature"] != signature(lock["protocol"]) or lock["protocol"]["experiment"] != experiment:
        raise ValueError("Invalid protocol signature or experiment")
    verify_inputs(lock["protocol"])
    return lock


def build_protocol(experiment: str, datasets: list[str], seeds: list[int] | None = None,
                   batch_size: int = 32, chunk_size: int = 256) -> dict:
    protocol = {"schema_version": 1, "experiment": experiment, "methods": METHODS,
                "training_performed": False, "inference_performed": experiment == "noise_sensitivity",
                "uored_test_opened": False, "new_independent_test_opened": False,
                "analysis_role": "supplementary sensitivity analysis of previously evaluated PU/CWRU tests",
                "source_vote_rule": "window majority; tied classes use float64 logit sums; remaining ties use lower class index",
                "retained_values": 512, "window_samples": 1024, "sampling_rate_hz": 12000,
                "runs": [], "inputs": [], "inference_cells": 0}
    if experiment == "noise_sensitivity":
        if batch_size < 1 or chunk_size < 1:
            raise ValueError("Batch and chunk sizes must be positive")
        protocol.update({"snr_db": [20.0, 10.0], "noise_master_seed": 20260930,
                         "noise_rule": "same zero-mean Gaussian realization per sample ID; exact power scaling before max_abs normalization",
                         "noise_power_reference": "original resampled observed window, not a noise-free ground truth",
                         "pca_refitted": False, "checkpoint_selection_changed": False,
                         "batch_size": batch_size, "chunk_size": chunk_size,
                         "inference_precision": "float32 without autocast; original export preprocessing"})
    implementation = [Path(__file__), Path(__file__).with_name(f"run_rie_{experiment}.py"),
                      Path(__file__).with_name("validate_rie_supplement.py")]
    if experiment == "noise_sensitivity":
        implementation += [ROOT / name for name in ["scripts/run_clean_study.py", "scripts/benchmark_followup_efficiency.py",
                           "src/reconstruction/compression_baselines.py", "src/reconstruction/sparse_recovery.py",
                           "src/data/pu_loader.py", "src/data/cwru_loader.py", "src/training/train_classifier.py",
                           "src/models/attention_resnet.py"]]
    protocol["inputs"].extend(artifact(path) for path in implementation)
    for dataset in datasets:
        benchmark = read_json(BENCHMARKS[dataset])
        source = read_json(SOURCES[dataset])
        if not benchmark["complete"] or source["benchmark_summary_sha256"] != digest(BENCHMARKS[dataset]):
            raise ValueError("Source analysis provenance differs from the formal benchmark")
        protocol["inputs"].extend([artifact(BENCHMARKS[dataset]), artifact(SOURCES[dataset])])
        protocol["inputs"].extend(benchmark_protocol_inputs(dataset, benchmark))
        selected_seeds = seeds if seeds is not None else sorted(map(int, benchmark["runs"]))
        for seed in selected_seeds:
            provenance = source["run_provenance"][str(seed)]
            run = {"dataset": dataset, "seed": seed, "methods": {},
                   "paper_label_names": {"ball": "compound"} if dataset == "pu" else {}}
            baseline = None
            for method in METHODS:
                formal = benchmark["runs"][str(seed)][method]
                info = provenance[method]
                entry = {"prediction": artifact(info["prediction_file"], info["prediction_sha256"]),
                         "manifest": artifact(info["manifest"], info["manifest_sha256"]),
                         "prediction_metadata": artifact(info["prediction_metadata"]),
                         "metrics": artifact(formal["metrics_path"], formal["metrics_sha256"])}
                metadata = read_json(info["prediction_metadata"])
                if metadata["files"]["test"]["sha256"] != entry["prediction"]["sha256"]:
                    raise ValueError("Prediction metadata checksum differs")
                data = arrays(entry["prediction"]["path"])
                manifest = read_json(entry["manifest"]["path"])
                if metadata["export_config"]["dataset_signature"] != manifest["signature"]:
                    raise ValueError("Prediction export and manifest signatures differ")
                if metadata["class_names"] != data["class_names"].tolist():
                    raise ValueError("Prediction export class order differs")
                records = ordered_records(manifest, data)
                if baseline is None:
                    baseline = data
                    baseline_records = [(r["source_file"], r["segment_id"], r["label"]) for r in records]
                else:
                    align(baseline, data)
                    if baseline_records != [(r["source_file"], r["segment_id"], r["label"]) for r in records]:
                        raise ValueError("Methods do not use identical source windows")
                if experiment == "noise_sensitivity":
                    condition = manifest["condition"]
                    expected_signal = {"segment_length": 1024, "stride": 1024, "sampling_rate": 12000, "normalize": "max_abs"}
                    if any(condition["signal"][key] != value for key, value in expected_signal.items()):
                        raise ValueError("Formal signal settings differ from the locked noise design")
                    if method != "no_cs" and condition["compression_ratio"] != 0.5:
                        raise ValueError("Expected formal 50% retention")
                    entry["checkpoint"] = artifact(formal["checkpoint"], metadata["export_config"]["checkpoint_sha256"])
                    if method == "pca":
                        pca_path = Path(formal["dataset_root"]) / "pca_training_only.npz"
                        pca_meta = pca_path.with_suffix(".json")
                        entry["pca_metadata"] = artifact(pca_meta)
                        metadata_pca = read_json(pca_meta)
                        if metadata_pca["fit_scope"] != "training_only":
                            raise ValueError("Formal PCA was not training-only")
                        training_ids = {row["sample_id"] for row in manifest["records"] if row["split"] == "train"}
                        if not set(metadata_pca["fit_sample_ids"]).issubset(training_ids):
                            raise ValueError("PCA fit IDs are not confined to the formal training subset")
                        entry["pca"] = artifact(pca_path, metadata_pca["sha256"])
                    indices = sorted(set([0, len(data["y_true"]) // 2, len(data["y_true"]) - 1]))
                    entry["clean_image_checks"] = [{"index": i, **artifact(records[i]["image_path"])} for i in indices]
                protocol["inputs"].extend(value for value in entry.values() if isinstance(value, dict) and "sha256" in value)
                protocol["inputs"].extend({"path": item["path"], "sha256": item["sha256"]} for item in entry.get("clean_image_checks", []))
                run["methods"][method] = entry
            groups = defaultdict(list)
            for row in records:
                groups[row["source_file"]].append(row)
            minimum = min(map(len, groups.values()))
            run.update({"test_windows": len(baseline["y_true"]), "source_files": len(groups),
                        "minimum_source_windows": minimum,
                        "window_lengths": [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= minimum] + ["full"],
                        "skipped_lengths": [k for k in [1, 2, 4, 8, 16, 32, 64] if k > minimum]})
            if experiment == "noise_sensitivity":
                declared = {str(Path(item["path"]).resolve()): item["sha256"] for item in manifest["condition"]["sources"]}
                run["raw_sources"] = [artifact(path, declared[str(Path(path).resolve())]) for path in sorted(groups)]
                protocol["inputs"].extend(run["raw_sources"])
                protocol["inference_cells"] += len(METHODS) * len(protocol["snr_db"])
            protocol["runs"].append(run)
    unique = {}
    for item in protocol["inputs"]:
        if item["path"] in unique and unique[item["path"]] != item["sha256"]:
            raise ValueError("Conflicting input checksums")
        unique[item["path"]] = item["sha256"]
    protocol["inputs"] = [{"path": path, "sha256": sha} for path, sha in sorted(unique.items())]
    return protocol


def scores(targets: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, recall_score
    labels = list(range(len(class_names)))
    recalls = recall_score(targets, predictions, labels=labels, average=None, zero_division=0)
    return {"macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(np.mean(targets == predictions)),
            "per_class_recall": dict(zip(class_names, map(float, recalls))),
            "minimum_class_recall": float(recalls.min()),
            "minimum_recall_classes": [class_names[i] for i in np.flatnonzero(recalls == recalls.min())],
            "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist()}


def evaluate(payload: dict, records: list[dict], length: int | str = "full") -> dict:
    names = payload["class_names"].tolist()
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row["source_file"]].append(i)
    source_rows, used = [], []
    for source, indices in sorted(groups.items()):
        indices = sorted(indices, key=lambda i: int(records[i]["segment_id"]))
        if length != "full":
            indices = indices[:int(length)]
        targets = payload["y_true"][indices]
        if np.any(targets != targets[0]):
            raise ValueError("A source file has multiple labels")
        logits = payload["logits"][indices]
        predicted = logits.argmax(axis=1)
        counts = np.bincount(predicted, minlength=len(names))
        tied = np.flatnonzero(counts == counts.max())
        sums = logits.astype(np.float64).sum(axis=0)
        winner = int(tied[np.argmax(sums[tied])])
        source_rows.append({"source_file": source, "target": int(targets[0]), "prediction": winner,
                            "class_name": names[int(targets[0])], "predicted_class": names[winner],
                            "window_count": len(indices), "window_accuracy": float(np.mean(predicted == targets)),
                            "correct": bool(winner == targets[0])})
        used.extend(indices)
    window = scores(payload["y_true"][used], payload["logits"][used].argmax(axis=1), names)
    source = scores(np.asarray([r["target"] for r in source_rows]), np.asarray([r["prediction"] for r in source_rows]), names)
    return {"window": window, "source": source, "failed_source_count": sum(not r["correct"] for r in source_rows),
            "source_file_count": len(source_rows), "window_count": len(used), "source_rows": source_rows}


def aggregate(cells: list[dict], keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for cell in cells:
        groups[tuple(cell[key] for key in keys)].append(cell)
    result = []
    for values, rows in groups.items():
        row = dict(zip(keys, values))
        row["seed_count"] = len(rows)
        for label, values in {
            "window_macro_f1": [r["metrics"]["window"]["macro_f1"] for r in rows],
            "minimum_class_recall": [r["metrics"]["window"]["minimum_class_recall"] for r in rows],
            "source_macro_f1": [r["metrics"]["source"]["macro_f1"] for r in rows],
            "failed_source_count": [r["metrics"]["failed_source_count"] for r in rows],
        }.items():
            row[label] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        result.append(row)
    return result
