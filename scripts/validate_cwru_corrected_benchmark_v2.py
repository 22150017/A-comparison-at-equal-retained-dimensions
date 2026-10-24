from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object, length: int = 24) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())[:length]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metrics(predictions: dict, records: list[dict]) -> dict:
    ids, targets, logits = predictions["sample_ids"].tolist(), predictions["y_true"], predictions["logits"]
    classes = predictions["class_names"].tolist()
    require(classes == ["ball", "inner_race", "normal", "outer_race"], "Class order differs")
    require(logits.shape == (len(targets), 4) and np.isfinite(logits).all(), "Invalid logits")
    require(len(ids) == len(set(ids)), "Duplicate prediction IDs")
    lookup = {r["sample_id"]: r for r in records if r["split"] == "test"}
    require(set(ids) == set(lookup), "Test prediction IDs differ")
    predicted = np.argmax(logits, axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    groups = {}
    for i, identity in enumerate(ids):
        require(classes[int(targets[i])] == lookup[identity]["label"], "Prediction target differs")
        confusion[int(targets[i]), int(predicted[i])] += 1
        groups.setdefault(lookup[identity]["source_file"], []).append(i)
    counts = confusion.sum(axis=1)
    require(np.all(counts > 0), "Missing test class")
    recall = np.diag(confusion) / counts
    denominator = confusion.sum(axis=1) + confusion.sum(axis=0)
    window_f1 = float(np.mean(np.divide(2 * np.diag(confusion), denominator, out=np.zeros(4), where=denominator != 0)))
    source_matrix, sources = np.zeros((4, 4), dtype=np.int64), []
    for source in sorted(groups):
        selected = groups[source]
        truth = set(int(targets[i]) for i in selected)
        require(len(truth) == 1, "Mixed source labels")
        votes = [sum(int(predicted[i]) == label for i in selected) for label in range(4)]
        tied = [label for label in range(4) if votes[label] == max(votes)]
        sums = {label: float(np.sum(logits[selected, label], dtype=np.float64)) for label in tied}
        winner = min(tied, key=lambda label: (-sums[label], label))
        target = truth.pop()
        source_matrix[target, winner] += 1
        sources.append({"source_file": source, "target": target, "prediction": winner, "window_count": len(selected),
                        "window_accuracy": float(sum(int(predicted[i]) == target for i in selected) / len(selected))})
    denominator = source_matrix.sum(axis=0) + source_matrix.sum(axis=1)
    source_f1 = float(np.mean(np.divide(2 * np.diag(source_matrix), denominator, out=np.zeros(4), where=denominator != 0)))
    return {"window_macro_f1": window_f1, "source_macro_f1": source_f1,
            "per_class_recall": dict(zip(classes, recall.tolist())), "minimum_class_recall": float(recall.min()),
            "confusion_matrix": confusion.tolist(), "sources": sources,
            "misclassified_sources": int(source_matrix.sum() - np.trace(source_matrix))}


def compare(actual: object, expected: object, name: str = "metrics") -> None:
    if isinstance(actual, dict):
        require(actual.keys() == expected.keys(), f"Keys differ: {name}")
        for key in actual:
            compare(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(actual, list):
        require(len(actual) == len(expected), f"Lengths differ: {name}")
        for a, b in zip(actual, expected):
            compare(a, b, name)
    elif isinstance(actual, float):
        require(abs(actual - expected) <= 1e-10, f"Value differs: {name}")
    else:
        require(actual == expected, f"Value differs: {name}")


def validate(path: Path, stage: str, seeds: list[int]) -> dict:
    lock = read_json(path)
    protocol = lock["protocol"]
    require(canonical(protocol) == lock["signature"], "Protocol signature differs")
    require(protocol["seeds"] == [42, 100, 200, 300, 400], "Seed set differs")
    require(set(seeds) <= set(protocol["seeds"]) and len(seeds) == len(set(seeds)), "Requested seeds differ")
    require(protocol["methods"] == ["no_cs", "ilsomp", "downsample", "pca", "random_projection"], "Method set differs")
    require(protocol["uored_signal_values_opened"] is False, "UORED access flag differs")
    for record in protocol["inputs"] + protocol["code"]:
        require(digest(Path(record["path"]).read_bytes()) == record["sha256"], f"Input changed: {record['path']}")
    config = protocol["config"]
    inventory = config["signal"]["cwru_source_inventory"]
    require(inventory["version"] == "cwru_explicit_de_resampled_v2", "Preprocessing version differs")
    source_root = Path(config["paths"]["cwru_root"])
    seen_channels, seen_windows, source_info, sample_labels = {}, {}, {}, {}
    normal_mapping = {"1797": "X097_DE_time", "1772": "X098_DE_time", "1750": "X099_DE_time", "1730": "X100_DE_time"}
    for row in inventory["sources"]:
        source = (source_root / row["relative_path"]).resolve()
        source.relative_to(source_root.resolve())
        require(digest(source.read_bytes()) == row["file_sha256"], f"Source SHA differs: {source}")
        if row["label"] == "normal":
            require(row["signal_key"] == normal_mapping[source.parent.name] and row["native_rate_hz"] == 48000, "Normal mapping differs")
        else:
            require(row["native_rate_hz"] == 12000 and "12DriveEndFault" in source.parts, "Fault collection differs")
        require(row["target_rate_hz"] == 12000, "Target sampling rate differs")
        raw = np.asarray(loadmat(source)[row["signal_key"]], dtype="<f8").reshape(-1)
        raw_hash = digest(raw.tobytes())
        require(raw_hash == row["raw_signal_sha256"] and raw.size == row["raw_samples"], "Selected signal differs")
        require(raw_hash not in seen_channels, "Duplicate selected raw channel")
        seen_channels[raw_hash] = str(source)
        vector = resample_poly(raw, 1, 4, window=("kaiser", 5.0), padtype="constant") if row["native_rate_hz"] == 48000 else raw
        require(vector.size == row["processed_samples"] and digest(np.asarray(vector, dtype="<f8").tobytes()) == row["processed_signal_sha256"], "Resampling differs")
        source_info[str(source)] = row
        for index, start in enumerate(range(0, len(vector) - 1024 + 1, 1024)):
            values = vector[start:start + 1024]
            values = values / (np.abs(values).max() + 1e-12)
            sha = digest(np.asarray(values, dtype="<f8").tobytes())
            require(sha not in seen_windows or seen_windows[sha] == str(source), "Duplicate normalized windows across files")
            seen_windows[sha] = str(source)
            identity = canonical({"source": str(source), "segment_id": index, "group_id": str(source)})
            require(identity not in sample_labels, "Duplicate sample ID")
            sample_labels[identity] = (row["label"], str(source))
    for seed in protocol["seeds"]:
        rows = protocol["split_rows"][str(seed)]
        require(len(rows) == len(sample_labels) and {r[0] for r in rows} == set(sample_labels), "Locked sample set differs")
        assignments, labels_by_role = {}, {role: set() for role in ("train", "val", "test")}
        for identity, label, role in rows:
            require(role in labels_by_role, "Unknown split")
            expected_label, source = sample_labels[identity]
            require(label == expected_label, "Locked target differs")
            require(source not in assignments or assignments[source] == role, "Source spans splits")
            assignments[source] = role
            labels_by_role[role].add(label)
        require(all(v == {"normal", "ball", "inner_race", "outer_race"} for v in labels_by_role.values()), "A split lacks classes")
    verified = 0
    if stage == "results":
        summary = read_json(path.parent / "corrected_benchmark_summary.json")
        require(summary["signature"] == lock["signature"] and summary["uored_signal_values_opened"] is False, "Summary metadata differs")
        require(summary["complete"] == (sum(len(v) for v in summary["runs"].values()) == 25), "Summary completion flag differs")
        for seed in seeds:
            paired = None
            for method in protocol["methods"]:
                cell = read_json(path.parent / "cells" / f"seed{seed}_{method}.json")
                require(cell == summary["runs"][str(seed)][method], "Summary cell differs")
                for entry in cell["artifacts"].values():
                    require(digest(Path(entry["path"]).read_bytes()) == entry["sha256"], "Cell artifact SHA differs")
                manifest = read_json(Path(cell["artifacts"]["manifest"]["path"]))
                require(manifest["complete"] is True, "Incomplete dataset")
                require(manifest["condition"]["signal"]["cwru_source_inventory"] == inventory, "Dataset preprocessing differs")
                rows = manifest["records"]
                require(sorted([r["sample_id"], r["label"], r["split"]] for r in rows) == protocol["split_rows"][str(seed)], "Materialized split differs")
                for row in rows:
                    require(row["signal_key"] == source_info[row["source_file"]]["signal_key"] and row["sampling_rate_hz"] == 12000, "Record channel/rate differs")
                with np.load(cell["artifacts"]["test_prediction"]["path"], allow_pickle=False) as data:
                    prediction = {key: data[key] for key in data.files}
                compare(metrics(prediction, rows), cell["metrics"])
                key = sorted(zip(prediction["sample_ids"].tolist(), prediction["y_true"].tolist()))
                require(paired is None or key == paired, "Paired method targets/IDs differ")
                paired = key
                trained = read_json(Path(cell["artifacts"]["metrics"]["path"]))
                require(trained["run_config"]["epochs"] == 30 and trained["checkpoint_selection"] == "best-val", "Training/selection rule differs")
                if method == "pca":
                    with np.load(cell["artifacts"]["pca"]["path"], allow_pickle=False) as data:
                        fit_ids = set(data["fit_sample_ids"].tolist())
                    training_ids = {r["sample_id"] for r in rows if r["split"] == "train"}
                    require(fit_ids <= training_ids and len(fit_ids) == min(4096, len(training_ids)), "PCA fit uses wrong samples")
                verified += 1
        for method, saved in summary["aggregate"].items():
            rows = [v[method]["metrics"] for v in summary["runs"].values() if method in v]
            for key, stats in saved.items():
                values = np.asarray([r[key] for r in rows])
                compare({"mean": float(values.mean()), "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
                         "n": len(values)}, stats, "aggregate")
    return {"valid": True, "stage": stage, "signature": lock["signature"], "verified_cells": verified,
            "selected_sources": len(source_info), "windows": len(sample_labels), "duplicate_selected_channels": 0,
            "duplicate_windows_across_sources": 0, "validator_training_performed": False,
            "validator_inference_performed": False, "uored_signal_values_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "results"), default="preflight")
    parser.add_argument("--seeds", default="42,100,200,300,400")
    args = parser.parse_args()
    print(json.dumps(validate(args.lock.resolve(), args.stage, [int(s) for s in args.seeds.split(",")]), indent=2))


if __name__ == "__main__":
    main()
