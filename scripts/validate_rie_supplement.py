from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from rie_supplement_support import METHODS, arrays, artifact, load_lock, read_json


def identical(observed, expected, location: str = "result") -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise ValueError(f"Fields differ: {location}")
        for key in expected:
            identical(observed[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"List lengths differ: {location}")
        for i, (left, right) in enumerate(zip(observed, expected)):
            identical(left, right, f"{location}[{i}]")
    elif isinstance(expected, float):
        if not np.isclose(observed, expected, rtol=0, atol=1e-12):
            raise ValueError(f"Numeric mismatch: {location}: {observed} != {expected}")
    elif observed != expected:
        raise ValueError(f"Value mismatch: {location}: {observed} != {expected}")


def pair_check(left: dict, right: dict) -> None:
    for key in ("sample_ids", "y_true", "class_names"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"Pairing differs: {key}")


def metric_from_confusion(y: np.ndarray, predicted: np.ndarray, names: list[str]) -> dict:
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for target, prediction in zip(y.tolist(), predicted.tolist()):
        matrix[int(target), int(prediction)] += 1
    true_positive = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    denominator = support + matrix.sum(axis=0)
    recall = np.divide(true_positive, support, out=np.zeros(len(names)), where=support != 0)
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(len(names)), where=denominator != 0)
    minimum = float(min(recall))
    return {"macro_f1": float(sum(f1) / len(names)), "accuracy": float(sum(true_positive) / len(y)),
            "per_class_recall": {name: float(recall[i]) for i, name in enumerate(names)},
            "minimum_class_recall": minimum,
            "minimum_recall_classes": [name for i, name in enumerate(names) if recall[i] == minimum],
            "confusion_matrix": matrix.tolist()}


def independent_metrics(data: dict, manifest: dict, length: int | str) -> dict:
    names = data["class_names"].tolist()
    record_list = [row for row in manifest["records"] if row["split"] == "test"]
    lookup = {row["sample_id"]: row for row in record_list}
    if len(lookup) != len(record_list) or set(lookup) != set(data["sample_ids"].tolist()):
        raise ValueError("Manifest test records do not uniquely match prediction IDs")
    source_map = defaultdict(list)
    for index, sample_id in enumerate(data["sample_ids"].tolist()):
        record = lookup[sample_id]
        if names[int(data["y_true"][index])] != record["label"]:
            raise ValueError("Manifest label differs from prediction target")
        source_map[record["source_file"]].append((int(record["segment_id"]), index))
    rows, selected = [], []
    for source in sorted(source_map):
        ordered = sorted(source_map[source])
        if len({segment for segment, _ in ordered}) != len(ordered):
            raise ValueError("Duplicated temporal position within a source")
        if length != "full":
            if len(ordered) < int(length):
                raise ValueError("Insufficient windows for the locked prefix length")
            ordered = ordered[:int(length)]
        indices = [i for _, i in ordered]
        targets = data["y_true"][indices]
        if len(set(targets.tolist())) != 1:
            raise ValueError("A source file has more than one class")
        predicted = np.argmax(data["logits"][indices], axis=1)
        votes = [int(np.count_nonzero(predicted == i)) for i in range(len(names))]
        maximum = max(votes)
        leaders = [i for i, count in enumerate(votes) if count == maximum]
        sums = data["logits"][indices].astype(np.float64).sum(axis=0)
        winner = max(leaders, key=lambda i: (float(sums[i]), -i))
        target = int(targets[0])
        rows.append({"source_file": source, "target": target, "prediction": winner, "class_name": names[target],
                     "predicted_class": names[winner], "window_count": len(indices),
                     "window_accuracy": float(np.count_nonzero(predicted == targets) / len(indices)),
                     "correct": winner == target})
        selected.extend(indices)
    window = metric_from_confusion(data["y_true"][selected], np.argmax(data["logits"][selected], axis=1), names)
    source = metric_from_confusion(np.array([r["target"] for r in rows]), np.array([r["prediction"] for r in rows]), names)
    return {"window": window, "source": source, "failed_source_count": len([r for r in rows if not r["correct"]]),
            "source_file_count": len(rows), "window_count": len(selected), "source_rows": rows}


def independent_aggregate(cells: list[dict], keys: list[str]) -> list[dict]:
    grouped = {}
    for row in cells:
        identity = tuple(row[key] for key in keys)
        grouped.setdefault(identity, []).append(row)
    output = []
    for identity, rows in grouped.items():
        item = dict(zip(keys, identity))
        item["seed_count"] = len(rows)
        sets = {"window_macro_f1": [r["metrics"]["window"]["macro_f1"] for r in rows],
                "minimum_class_recall": [r["metrics"]["window"]["minimum_class_recall"] for r in rows],
                "source_macro_f1": [r["metrics"]["source"]["macro_f1"] for r in rows],
                "failed_source_count": [r["metrics"]["failed_source_count"] for r in rows]}
        for name, values in sets.items():
            mean = sum(values) / len(values)
            sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else None
            item[name] = {"mean": float(mean), "std": None if sd is None else float(sd)}
        output.append(item)
    return output


def verify(lock_path: Path) -> None:
    header = read_json(lock_path)
    experiment = header["protocol"]["experiment"]
    if experiment not in {"record_length", "noise_sensitivity"}:
        raise ValueError("Unsupported supplement experiment")
    lock = load_lock(lock_path, experiment)
    protocol = lock["protocol"]
    if protocol["training_performed"] or protocol["uored_test_opened"] or protocol["new_independent_test_opened"]:
        raise ValueError("Supplement protocol violates scope")
    if protocol["methods"] != METHODS or any(run["dataset"] not in {"pu", "cwru"} for run in protocol["runs"]):
        raise ValueError("Unexpected dataset or method")
    if protocol["retained_values"] != 512 or protocol["window_samples"] != 1024 or protocol["sampling_rate_hz"] != 12000:
        raise ValueError("Unexpected representation settings")
    if protocol["inference_performed"] != (experiment == "noise_sensitivity"):
        raise ValueError("Inference scope flag differs")
    root = lock_path.resolve().parent
    summary = read_json(root / f"{experiment}_summary.json")
    if not summary["complete"] or summary["signature"] != lock["signature"]:
        raise ValueError("Incomplete or mismatched summary")
    identical(summary["protocol_lock"], artifact(lock_path))
    if summary["training_performed"] or summary["uored_test_opened"] or summary["new_independent_test_opened"] or summary["inference_performed"] != (experiment == "noise_sensitivity"):
        raise ValueError("Summary scope flags differ")
    actual_cells = summary["cells"]
    indexed = {}
    dimension = "length" if experiment == "record_length" else "snr_db"
    for cell in actual_cells:
        key = (cell["dataset"], cell["seed"], cell["method"], cell[dimension])
        if key in indexed:
            raise ValueError("Duplicate experiment cell")
        indexed[key] = cell
    recomputed, seen_noise, expected_count = [], {}, 0
    for run in protocol["runs"]:
        baseline = arrays(run["methods"]["no_cs"]["prediction"]["path"])
        for method in METHODS:
            entry = run["methods"][method]
            clean = arrays(entry["prediction"]["path"])
            pair_check(baseline, clean)
            manifest = read_json(entry["manifest"]["path"])
            reference = independent_metrics(clean, manifest, "full")
            formal_benchmark = read_json(next(item["path"] for item in protocol["inputs"]
                                             if Path(item["path"]).name == ("formal_pu_598ba4f57533501db61bcb0c_summary.json" if run["dataset"] == "pu" else "external_cwru_five_seed_d3be8e689220c8c902d1e2d3_summary.json")))
            formal_f1 = formal_benchmark["runs"][str(run["seed"])][method]["paper_metrics"]["macro_f1"]
            if not np.isclose(reference["window"]["macro_f1"], formal_f1, atol=1e-10, rtol=0):
                raise ValueError("Formal prediction F1 differs from paper benchmark")
            counts = [row["window_count"] for row in reference["source_rows"]]
            lengths = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= min(counts)] + ["full"]
            identical(run["window_lengths"], lengths)
            for setting in lengths if experiment == "record_length" else protocol["snr_db"]:
                key = (run["dataset"], run["seed"], method, setting)
                if key not in indexed:
                    raise ValueError(f"Missing cell: {key}")
                stored = indexed[key]
                if experiment == "record_length":
                    metrics = independent_metrics(clean, manifest, setting)
                    expected_time = None if setting == "full" else setting * 1024 / 12000
                    identical(stored["observation_seconds"], expected_time)
                else:
                    if stored["signature"] != lock["signature"] or stored["training_performed"] or stored["pca_refitted"] or stored["uored_test_opened"]:
                        raise ValueError("Noise cell scope differs")
                    identical(stored["checkpoint"], entry["checkpoint"])
                    artifact(stored["prediction"]["path"], stored["prediction"]["sha256"])
                    data = arrays(stored["prediction"]["path"])
                    pair_check(clean, data)
                    if data["actual_added_snr_db"].shape != clean["y_true"].shape or not np.allclose(data["actual_added_snr_db"], setting, atol=1e-10, rtol=0):
                        raise ValueError("Added-noise power ratio does not match locked SNR")
                    hashes = data["noisy_input_sha256"]
                    if hashes.shape != clean["y_true"].shape or any(len(str(x)) != 64 for x in hashes):
                        raise ValueError("Invalid noisy input fingerprints")
                    identity = (run["dataset"], run["seed"], setting)
                    if identity in seen_noise and not np.array_equal(seen_noise[identity], hashes):
                        raise ValueError("Methods did not receive identical noisy inputs")
                    seen_noise[identity] = hashes
                    checks = [{"sample_id": str(clean["sample_ids"][row["index"]]), "pixels_equal": True}
                              for row in entry["clean_image_checks"]]
                    identical(stored["clean_frontend_checks"], checks)
                    metrics = independent_metrics(data, manifest, "full")
                    identical(stored["clean_reference"], reference)
                    delta = {"window_macro_f1": metrics["window"]["macro_f1"] - reference["window"]["macro_f1"],
                             "minimum_class_recall": metrics["window"]["minimum_class_recall"] - reference["window"]["minimum_class_recall"],
                             "source_macro_f1": metrics["source"]["macro_f1"] - reference["source"]["macro_f1"],
                             "failed_source_count": metrics["failed_source_count"] - reference["failed_source_count"],
                             "per_class_recall": {name: metrics["window"]["per_class_recall"][name] - reference["window"]["per_class_recall"][name]
                                                  for name in clean["class_names"].tolist()}}
                    identical(stored["delta_vs_clean"], delta)
                identical(stored["metrics"], metrics)
                recomputed.append({"dataset": run["dataset"], "seed": run["seed"], "method": method, dimension: setting, "metrics": metrics})
                expected_count += 1
    if len(indexed) != expected_count:
        raise ValueError("Unexpected extra cells")
    identical(summary["aggregate"], independent_aggregate(recomputed, ["dataset", "method", dimension]))
    if experiment == "record_length":
        lookup = {(row["dataset"], row["seed"], row["length"]): row for row in recomputed if row["method"] == "no_cs"}
        paired = []
        for row in recomputed:
            if row["method"] == "no_cs":
                continue
            ref = lookup[(row["dataset"], row["seed"], row["length"])]["metrics"]
            paired.append({key: row[key] for key in ("dataset", "seed", "method", "length")} | {
                "source_macro_f1_delta_vs_no_cs": row["metrics"]["source"]["macro_f1"] - ref["source"]["macro_f1"],
                "failed_source_count_delta_vs_no_cs": row["metrics"]["failed_source_count"] - ref["failed_source_count"]})
        identical(summary["paired_comparisons"], paired)
    else:
        if expected_count != protocol["inference_cells"] or summary["pca_refitted"] or summary["new_independent_test_opened"]:
            raise ValueError("Noise cell count or final scope differs")
    print(f"OK: independently validated {expected_count} {experiment} cells, metrics and paired inputs; no training or UORED test access")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    verify(args.lock)


if __name__ == "__main__":
    main()
