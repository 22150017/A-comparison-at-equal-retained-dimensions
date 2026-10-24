





from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_split_predictions import export_split_predictions
from src.utils.io_tools import export_json
from src.utils.reproducibility import canonical_hash, file_sha256

DEFAULT_PROTOCOL_SIGNATURE = "598ba4f57533501db61bcb0c"
STAGE_FILENAMES = {
    "formal-pu": "formal_pu_{signature}_summary.json",
    "external-cwru": "external_cwru_{signature}_summary.json",
}


def _summary_path(stage: str, signature: str) -> Path:
    return (
        ROOT
        / "artifacts_v2"
        / "studies"
        / "benchmark"
        / STAGE_FILENAMES[stage].format(signature=signature)
    )


def _validate_benchmark(payload: dict, stage: str) -> None:
    if not payload.get("complete") or not payload.get("paper_claim_authorized"):
        raise ValueError(f"complete paper-authorized benchmark required for {stage}")
    allowed_stages = {stage}
    if stage == "external-cwru":
        allowed_stages.add("external-cwru-five-seed-extension")
    if payload.get("stage") not in allowed_stages:
        raise ValueError(
            f"benchmark stage mismatch: expected one of {sorted(allowed_stages)}, "
            f"got {payload.get('stage')}"
        )
    design = payload.get("stage_design", {})
    if design.get("split_unit") != "source-file" or design.get("evaluation_scope") != "formal-test":
        raise ValueError("source-file analysis requires a formal source-file benchmark")
    expected = {
        (str(seed), method)
        for seed in design.get("seeds", [])
        for method in design.get("methods", [])
    }
    observed = {
        (seed, method)
        for seed, methods in payload.get("runs", {}).items()
        for method in methods
    }
    if observed != expected:
        raise ValueError(f"benchmark cells incomplete: {len(observed)}/{len(expected)}")


def _prediction_metadata(run: dict, backbone: str) -> tuple[Path, dict]:
    metadata_path = export_split_predictions(
        Path(run["dataset_root"]),
        Path(run["result_root"]),
        backbone=backbone,
        batch_size=64,
        eval_num_workers=0,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    test_file = metadata.get("files", {}).get("test", {})
    prediction_path = metadata_path.parent / str(test_file.get("name", ""))
    if not prediction_path.is_file() or file_sha256(prediction_path) != test_file.get("sha256"):
        raise RuntimeError(f"invalid test prediction sidecar: {prediction_path}")
    return prediction_path, metadata


def _majority_prediction(predictions: np.ndarray, logits: np.ndarray, class_count: int) -> int:
    counts = np.bincount(predictions, minlength=class_count)
    leaders = np.flatnonzero(counts == counts.max())
    if len(leaders) == 1:
        return int(leaders[0])
    scores = logits[:, leaders].sum(axis=0)
    return int(leaders[int(np.argmax(scores))])


def source_rows_from_predictions(
    manifest: dict,
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
) -> tuple[list[dict], dict]:
    
    if logits.ndim != 2 or logits.shape != (len(sample_ids), len(class_names)):
        raise ValueError("prediction logits have an incompatible shape")
    if len(y_true) != len(sample_ids):
        raise ValueError("prediction labels and sample IDs have different lengths")
    test_records = [record for record in manifest.get("records", []) if record.get("split") == "test"]
    records_by_id = {str(record.get("sample_id")): record for record in test_records}
    if len(records_by_id) != len(test_records):
        raise ValueError("test manifest contains duplicate sample IDs")
    ids = [str(value) for value in sample_ids.tolist()]
    if len(ids) != len(set(ids)) or set(ids) != set(records_by_id):
        raise ValueError("prediction sample IDs do not match the complete manifest test split")
    label_to_index = {name: index for index, name in enumerate(class_names)}
    predictions = np.asarray(logits.argmax(axis=1), dtype=np.int64)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(ids):
        record = records_by_id[sample_id]
        label = str(record.get("label"))
        if label not in label_to_index or int(y_true[index]) != label_to_index[label]:
            raise ValueError(f"label mismatch for sample {sample_id}")
        source_file = str(record.get("source_file", ""))
        if not source_file:
            raise ValueError(f"missing source file for sample {sample_id}")
        groups[source_file].append(index)

    rows: list[dict] = []
    for source_file in sorted(groups):
        indices = np.asarray(groups[source_file], dtype=np.int64)
        true_values = np.unique(y_true[indices])
        if len(true_values) != 1:
            raise ValueError(f"source file has multiple true labels: {source_file}")
        true_index = int(true_values[0])
        predicted = predictions[indices]
        majority_index = _majority_prediction(predicted, logits[indices], len(class_names))
        counts = np.bincount(predicted, minlength=len(class_names))
        source_id = canonical_hash(source_file, length=16)
        rows.append({
            "source_file": source_file,
            "source_name": Path(source_file).name,
            "source_id": source_id,
            "source_label": f"{Path(source_file).name} [{source_id[:8]}]",
            "true_label": class_names[true_index],
            "window_count": int(len(indices)),
            "correct_window_count": int(np.sum(predicted == true_index)),
            "window_accuracy": float(np.mean(predicted == true_index)),
            "majority_predicted_label": class_names[majority_index],
            "majority_correct": bool(majority_index == true_index),
            "prediction_counts": {
                class_name: int(counts[index]) for index, class_name in enumerate(class_names)
            },
        })

    recomputed = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=np.arange(len(class_names)), average="macro", zero_division=0)),
        "sample_count": int(len(y_true)),
    }
    return rows, recomputed


def _macro_source_window_accuracy(rows: list[dict], class_names: list[str]) -> float:
    class_means = []
    for class_name in class_names:
        values = [float(row["window_accuracy"]) for row in rows if row["true_label"] == class_name]
        if not values:
            raise ValueError(f"source-level rows lack class {class_name}")
        class_means.append(float(np.mean(values)))
    return float(np.mean(class_means))


def _stratified_bootstrap_ci(
    rows: list[dict],
    class_names: list[str],
    seed: int,
    repeats: int = 2000,
) -> list[float]:
    by_class = {
        class_name: [row for row in rows if row["true_label"] == class_name]
        for class_name in class_names
    }
    if any(not values for values in by_class.values()):
        raise ValueError("stratified source bootstrap requires every class")
    generator = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = []
        for class_name in class_names:
            class_rows = by_class[class_name]
            selected = generator.integers(0, len(class_rows), size=len(class_rows))
            sampled.extend(class_rows[int(position)] for position in selected)
        values[index] = _macro_source_window_accuracy(sampled, class_names)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def aggregate_source_rows(
    rows: list[dict],
    class_names: list[str],
    bootstrap_seed: int,
) -> dict:
    y_true = np.asarray([class_names.index(row["true_label"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray(
        [class_names.index(row["majority_predicted_label"]) for row in rows], dtype=np.int64
    )
    per_class_recall = {}
    per_class_mean_window_accuracy = {}
    for index, class_name in enumerate(class_names):
        mask = y_true == index
        if not np.any(mask):
            raise ValueError(f"source-level aggregate lacks class {class_name}")
        per_class_recall[class_name] = float(np.mean(y_pred[mask] == index))
        per_class_mean_window_accuracy[class_name] = float(
            np.mean([row["window_accuracy"] for row in rows if row["true_label"] == class_name])
        )
    window_accuracies = np.asarray([row["window_accuracy"] for row in rows], dtype=np.float64)
    macro_source_accuracy = _macro_source_window_accuracy(rows, class_names)
    return {
        "source_file_count": len(rows),
        "window_count": int(sum(int(row["window_count"]) for row in rows)),
        "mean_source_window_accuracy": float(window_accuracies.mean()),
        "std_source_window_accuracy": float(window_accuracies.std(ddof=1)) if len(rows) > 1 else None,
        "median_source_window_accuracy": float(np.median(window_accuracies)),
        "minimum_source_window_accuracy": float(window_accuracies.min()),
        "macro_source_window_accuracy": macro_source_accuracy,
        "macro_source_window_accuracy_bootstrap_95ci": _stratified_bootstrap_ci(
            rows, class_names, bootstrap_seed
        ),
        "majority_vote_accuracy": float(accuracy_score(y_true, y_pred)),
        "majority_vote_macro_f1": float(
            f1_score(y_true, y_pred, labels=np.arange(len(class_names)), average="macro", zero_division=0)
        ),
        "majority_vote_per_class_recall": per_class_recall,
        "per_class_mean_source_window_accuracy": per_class_mean_window_accuracy,
        "failed_source_files": [
            row.get("source_label", f"{row['source_name']} [{row.get('source_id', 'unknown')[:8]}]")
            for row in rows if not row["majority_correct"]
        ],
        "low_accuracy_source_files": [
            row.get("source_label", f"{row['source_name']} [{row.get('source_id', 'unknown')[:8]}]")
            for row in rows if float(row["window_accuracy"]) < 0.5
        ],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = []
    for row in rows:
        item = dict(row)
        item["prediction_counts"] = json.dumps(item["prediction_counts"], sort_keys=True)
        flattened.append(item)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def _write_markdown(path: Path, stage: str, aggregate: dict) -> None:
    lines = [
        f"# Source-file performance: {stage}",
        "",
        "Source files are the descriptive analysis units. Windows remain nested observations.",
        "",
        "| Seed | Method | Sources | Macro source accuracy | 95% bootstrap CI | Majority Macro-F1 | Failed sources |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed, methods in aggregate.items():
        for method, item in methods.items():
            interval = item["macro_source_window_accuracy_bootstrap_95ci"]
            lines.append(
                f"| {seed} | {method} | {item['source_file_count']} | "
                f"{item['macro_source_window_accuracy']:.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
                f"{item['majority_vote_macro_f1']:.4f} | {len(item['failed_source_files'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_stage(
    stage: str,
    protocol_signature: str,
    output_root: Path,
    benchmark_path: Path | None = None,
) -> dict:
    benchmark_path = benchmark_path or _summary_path(stage, protocol_signature)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    _validate_benchmark(benchmark, stage)
    analysis_version = canonical_hash(
        {
            "schema_version": 1,
            "stage": stage,
            "benchmark_summary_sha256": file_sha256(benchmark_path),
            "bootstrap_repeats": 2000,
        },
        length=12,
    )
    backbone = "resnet34_ema"
    all_rows: list[dict] = []
    aggregate: dict[str, dict] = {}
    run_provenance: dict[str, dict] = {}
    total_runs = sum(len(methods) for methods in benchmark["runs"].values())
    completed = 0
    for seed, methods in benchmark["runs"].items():
        aggregate[seed] = {}
        run_provenance[seed] = {}
        for method, run in methods.items():
            completed += 1
            print(f"[source-file {stage} {completed}/{total_runs}] seed={seed}, method={method}")
            prediction_path, metadata = _prediction_metadata(run, backbone)
            manifest_path = Path(run["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with np.load(prediction_path, allow_pickle=False) as predictions:
                sample_ids = predictions["sample_ids"]
                y_true = predictions["y_true"].astype(np.int64)
                logits = predictions["logits"].astype(np.float64)
                class_names = [str(value) for value in predictions["class_names"].tolist()]
            rows, recomputed = source_rows_from_predictions(
                manifest, sample_ids, y_true, logits, class_names
            )
            expected = run["paper_metrics"]
            for metric in ("accuracy", "macro_f1"):
                if not np.isclose(recomputed[metric], float(expected[metric]), rtol=0, atol=1e-10):
                    raise RuntimeError(f"{stage}/{seed}/{method} prediction {metric} mismatch")
            if recomputed["sample_count"] != int(expected["sample_count"]):
                raise RuntimeError(f"{stage}/{seed}/{method} prediction sample count mismatch")
            for row in rows:
                row.update({"stage": stage, "seed": int(seed), "method": method})
            all_rows.extend(rows)
            aggregate[seed][method] = aggregate_source_rows(
                rows, class_names, bootstrap_seed=int(seed) + 20260816
            )
            run_provenance[seed][method] = {
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "prediction_metadata": str(prediction_path.parent / f"{backbone}_split_predictions_metadata.json"),
                "prediction_file": str(prediction_path),
                "prediction_sha256": file_sha256(prediction_path),
                "prediction_export_signature": metadata["export_signature"],
            }
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{stage}_{analysis_version}_source_file"
    csv_path = output_root / f"{stem}_rows.csv"
    markdown_path = output_root / f"{stem}_summary.md"
    summary_path = output_root / f"{stem}_summary.json"
    _write_csv(csv_path, all_rows)
    _write_markdown(markdown_path, stage, aggregate)
    summary = {
        "schema_version": 1,
        "complete": True,
        "stage": stage,
        "analysis_version": analysis_version,
        "protocol_signature": protocol_signature,
        "benchmark_stage": benchmark.get("stage"),
        "benchmark_summary": str(benchmark_path.resolve()),
        "benchmark_summary_sha256": file_sha256(benchmark_path),
        "analysis_unit": "source_file",
        "nested_observation_unit": "window",
        "bootstrap": {
            "method": "stratified source-file resampling within true class",
            "repeats": 2000,
            "interval": 0.95,
        },
        "claim_boundary": (
            "Source-file rows support grouped robustness descriptions. Windows within a source "
            "file are not independent inferential samples; bootstrap intervals resample source files."
        ),
        "aggregate": aggregate,
        "run_provenance": run_provenance,
        "files": {
            "source_rows_csv": str(csv_path),
            "source_rows_csv_sha256": file_sha256(csv_path),
            "summary_markdown": str(markdown_path),
            "summary_markdown_sha256": file_sha256(markdown_path),
        },
    }
    export_json(summary_path, summary)
    return {"summary": str(summary_path), "complete": True, "runs": total_runs}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stage", choices=("formal-pu", "external-cwru", "all"), default="all")
    cli.add_argument("--protocol-signature", default=DEFAULT_PROTOCOL_SIGNATURE)
    cli.add_argument(
        "--cwru-summary",
        type=Path,
        help="optional completed five-seed CWRU extension summary; defaults to the locked three-seed summary",
    )
    cli.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts_v2" / "reports" / "source_file_performance",
    )
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    stages = list(STAGE_FILENAMES) if args.stage == "all" else [args.stage]
    plans = []
    for stage in stages:
        path = (
            args.cwru_summary
            if stage == "external-cwru" and args.cwru_summary is not None
            else _summary_path(stage, args.protocol_signature)
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_benchmark(payload, stage)
        plans.append({
            "stage": stage,
            "benchmark_summary": str(path),
            "run_count": sum(len(methods) for methods in payload["runs"].values()),
            "output_version": canonical_hash(
                {
                    "schema_version": 1,
                    "stage": stage,
                    "benchmark_summary_sha256": file_sha256(path),
                    "bootstrap_repeats": 2000,
                },
                length=12,
            ),
        })
    if args.dry_run:
        print(json.dumps({
            "status": "ready",
            "plans": plans,
            "trains_models": False,
            "reuses_saved_checkpoints": True,
            "analysis_unit": "source_file",
        }, indent=2, ensure_ascii=False))
        return
    results = [
        build_stage(
            stage,
            args.protocol_signature,
            args.output_root,
            benchmark_path=(
                args.cwru_summary
                if stage == "external-cwru" and args.cwru_summary is not None
                else None
            ),
        )
        for stage in stages
    ]
    print(json.dumps({"status": "complete", "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
