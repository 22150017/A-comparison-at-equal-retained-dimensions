from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_clean_study import _sample_identity, load_config, run_condition
from scripts.run_leakage_aware_benchmark import _condition_args
from src.data.cwru_corrected import build_inventory, load_corrected_segments
from src.utils.io_tools import export_json, stratified_group_split_three_way
from src.utils.reproducibility import canonical_hash, file_sha256

DEFAULT_ROOT = Path("F:/paper1/artifacts_v2/studies/cwru_corrected_benchmark_v2")
SEEDS = [42, 100, 200, 300, 400]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def split_rows(samples: list, config: dict, seed: int) -> list[list]:
    indices = stratified_group_split_three_way([s.label for s in samples], [s.group_id for s in samples],
                                              config["training"]["train_ratio"], config["training"]["val_ratio"], seed)
    return sorted([_sample_identity(samples[int(i)]), samples[int(i)].label, role]
                  for role, selected in zip(("train", "val", "test"), indices) for i in selected)


def build_protocol() -> dict:
    base_path = ROOT / "configs/leakage_aware_benchmark_v1.json"
    config_path = ROOT / "configs/default_config.yaml"
    base, config = read_json(base_path), load_config(config_path)
    config = deepcopy(config)
    inventory = build_inventory(Path(config["paths"]["cwru_root"]), config["signal"]["allowed_path_keywords"])
    config["signal"]["cwru_source_inventory"] = inventory
    samples = load_corrected_segments(Path(config["paths"]["cwru_root"]), inventory)
    splits = {str(seed): split_rows(samples, config, seed) for seed in SEEDS}
    code = [Path(__file__), ROOT / "scripts/run_clean_study.py", ROOT / "scripts/run_leakage_aware_benchmark.py",
            ROOT / "src/data/cwru_corrected.py", ROOT / "src/data/cwru_loader.py", ROOT / "src/data/pu_loader.py",
            ROOT / "src/utils/io_tools.py", ROOT / "src/utils/reproducibility.py",
            ROOT / "src/reconstruction/sparse_recovery.py", ROOT / "src/reconstruction/compression_baselines.py",
            ROOT / "src/models/attention_resnet.py", ROOT / "src/training/train_classifier.py",
            ROOT / "scripts/export_split_predictions.py", ROOT / "scripts/run_rie_noise_sensitivity.py",
            ROOT / "scripts/validate_cwru_corrected_benchmark_v2.py"]
    return {"schema_version": 2, "experiment": "cwru_corrected_benchmark_v2", "config": config,
            "base_protocol": base, "inputs": [artifact(base_path), artifact(config_path)],
            "code": [artifact(p) for p in code], "seeds": SEEDS, "methods": base["methods"],
            "split_rows": splits, "training_cells": 25, "new_independent_confirmation": False,
            "reason": "Correct DE record identity, resample normal baseline from 48kHz to 12kHz, reject duplicate channel/window content",
            "post_training_prediction_exports": ["val", "test"], "uored_signal_values_opened": False,
            "timing": "New classification results require no reuse of old CWRU model/PCA/predictions",
            "noise_and_prefix_results": "Recompute after corrected clean benchmark validation; old derived CWRU results are not merged"}


def load_lock(path: Path) -> dict:
    lock = read_json(path)
    if canonical_hash(lock["protocol"], length=24) != lock["signature"]:
        raise ValueError("Protocol signature differs")
    for record in lock["protocol"]["inputs"] + lock["protocol"]["code"]:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Locked input changed: {record['path']}")
    return lock


def summarize(prediction_path: Path, manifest: dict) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score

    with np.load(prediction_path, allow_pickle=False) as data:
        logits, targets = data["logits"], data["y_true"]
        classes, ids = data["class_names"].tolist(), data["sample_ids"].tolist()
    predicted = logits.argmax(axis=1)
    labels = list(range(len(classes)))
    matrix = confusion_matrix(targets, predicted, labels=labels)
    records = {r["sample_id"]: r for r in manifest["records"] if r["split"] == "test"}
    if set(ids) != set(records) or len(ids) != len(records):
        raise ValueError("Prediction IDs differ from test manifest")
    groups = {}
    for i, identity in enumerate(ids):
        record = records[identity]
        if classes[int(targets[i])] != record["label"]:
            raise ValueError("Prediction target differs from manifest")
        groups.setdefault(record["source_file"], []).append(i)
    sources = []
    for source, selected in sorted(groups.items()):
        idx = np.asarray(selected)
        truth = np.unique(targets[idx])
        if len(truth) != 1:
            raise ValueError("A source has multiple targets")
        counts = np.bincount(predicted[idx], minlength=len(classes))
        tied = np.flatnonzero(counts == counts.max())
        sums = logits[idx].astype(np.float64).sum(axis=0)
        winner = int(tied[np.argmax(sums[tied])])
        sources.append({"source_file": source, "target": int(truth[0]), "prediction": winner,
                        "window_count": len(idx), "window_accuracy": float(np.mean(predicted[idx] == targets[idx]))})
    recall = np.diag(matrix) / matrix.sum(axis=1)
    return {"window_macro_f1": float(f1_score(targets, predicted, labels=labels, average="macro", zero_division=0)),
            "per_class_recall": dict(zip(classes, recall.tolist())), "minimum_class_recall": float(recall.min()),
            "confusion_matrix": matrix.tolist(), "source_macro_f1": float(f1_score(
                [s["target"] for s in sources], [s["prediction"] for s in sources], labels=labels, average="macro", zero_division=0)),
            "misclassified_sources": sum(s["target"] != s["prediction"] for s in sources), "sources": sources}


def run(lock_path: Path, selected_seeds: list[int]) -> None:
    from scripts.export_split_predictions import export_split_predictions

    lock = load_lock(lock_path)
    protocol, root = lock["protocol"], lock_path.parent
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds) or set(selected_seeds) - set(SEEDS):
        raise ValueError("Choose unique locked seeds from 42,100,200,300,400")
    config = protocol["config"]
    inventory = config["signal"]["cwru_source_inventory"]
    samples = load_corrected_segments(Path(config["paths"]["cwru_root"]), inventory)
    for seed in selected_seeds:
        if split_rows(samples, config, seed) != protocol["split_rows"][str(seed)]:
            raise ValueError("Split differs from locked data")
    design = {"dataset": "cwru", "split_unit": "source-file", "evaluation_scope": "formal-test"}
    summary_path = root / "corrected_benchmark_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {
        "signature": lock["signature"], "complete": False, "runs": {}, "independent_confirmation": False,
        "uored_signal_values_opened": False}
    if summary["signature"] != lock["signature"]:
        raise ValueError("Summary belongs to another protocol")
    for seed in selected_seeds:
        for method in protocol["methods"]:
            cell = root / "cells" / f"seed{seed}_{method}.json"
            if cell.exists():
                payload = read_json(cell)
                if payload["signature"] != lock["signature"]:
                    raise ValueError("Cell protocol differs")
                for record in payload["artifacts"].values():
                    if file_sha256(Path(record["path"])) != record["sha256"]:
                        raise ValueError(f"Completed artifact changed: {record['path']}")
                print(f"[seed{seed}_{method}] complete; skipping", flush=True)
            else:
                args = _condition_args(protocol["base_protocol"], design, method, seed)
                args.artifact_root, args.reuse_artifact_roots = root / "artifacts_v2", ()
                raw = run_condition(args, config)
                data_root, result_root = Path(raw["dataset_root"]), Path(raw["result_root"])
                manifest_path = data_root / "dataset_manifest.json"
                manifest = read_json(manifest_path)
                actual = sorted([r["sample_id"], r["label"], r["split"]] for r in manifest["records"])
                if actual != protocol["split_rows"][str(seed)]:
                    raise ValueError("Materialized split differs from lock")
                metadata_path = export_split_predictions(data_root, result_root, args.backbone, batch_size=64,
                                                         eval_num_workers=0, splits=("val", "test"))
                metadata = read_json(metadata_path)
                prediction = result_root / metadata["files"]["test"]["name"]
                artifacts = {"manifest": artifact(manifest_path), "prediction_metadata": artifact(metadata_path),
                             "test_prediction": artifact(prediction), "checkpoint": artifact(Path(raw["checkpoint"])),
                             "val_prediction": artifact(result_root / metadata["files"]["val"]["name"]),
                             "metrics": artifact(result_root / f"{args.backbone}_metrics.json")}
                if method == "pca":
                    artifacts["pca"] = artifact(data_root / "pca_training_only.npz")
                    artifacts["pca_metadata"] = artifact(data_root / "pca_training_only.json")
                payload = {"signature": lock["signature"], "seed": seed, "method": method,
                           "artifacts": artifacts, "metrics": summarize(prediction, manifest)}
                export_json(cell, payload)
            summary["runs"].setdefault(str(seed), {})[method] = payload
            summary["complete"] = sum(len(v) for v in summary["runs"].values()) == 25
            summary["aggregate"] = {}
            for name in protocol["methods"]:
                rows = [v[name]["metrics"] for v in summary["runs"].values() if name in v]
                if rows:
                    summary["aggregate"][name] = {key: {
                        "mean": float(np.mean([r[key] for r in rows])),
                        "sample_standard_deviation": float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None,
                        "n": len(rows)} for key in ("window_macro_f1", "source_macro_f1", "minimum_class_recall")}
            export_json(summary_path, summary)
    print(json.dumps({"complete": summary["complete"], "completed_cells": sum(len(v) for v in summary["runs"].values()),
                      "summary": str(summary_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-signature")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", default="42")
    args = parser.parse_args()
    if args.stage == "run":
        if args.lock is None or args.dry_run:
            parser.error("run requires --lock and does not accept --dry-run")
        run(args.lock.resolve(), [int(s) for s in args.seeds.split(",")])
        return
    protocol = build_protocol()
    signature = canonical_hash(protocol, length=24)
    destination = args.output_root.resolve() / signature / "protocol_lock.json"
    preview = {"signature": signature, "lock": str(destination), "training_cells": 25,
               "first_run_seed42_cells": 5, "source_files": len(protocol["config"]["signal"]["cwru_source_inventory"]["sources"]),
               "windows": len(protocol["split_rows"]["42"]), "target_sampling_rate_hz": 12000,
               "normal_record_mapping": {s["relative_path"]: s["signal_key"] for s in protocol["config"]["signal"]["cwru_source_inventory"]["sources"] if s["label"] == "normal"},
               "training_performed": False, "inference_performed": False, "uored_signal_values_opened": False}
    if not args.dry_run:
        if args.confirm_signature != signature:
            parser.error("prepare requires the signature returned by --dry-run")
        if destination.exists():
            if read_json(destination) != {"signature": signature, "protocol": protocol}:
                raise ValueError("An incompatible lock already exists")
        else:
            export_json(destination, {"signature": signature, "protocol": protocol})
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
