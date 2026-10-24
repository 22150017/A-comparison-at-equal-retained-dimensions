
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import file_sha256
from scripts.validation_math import confusion_errors, statistic_errors


def validate_source_summary(path: Path) -> list[str]:
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete") or payload.get("analysis_unit") != "source_file":
        errors.append(f"incomplete/non-source-file analysis: {path}")
    if payload.get("nested_observation_unit") != "window" or not payload.get("claim_boundary"):
        errors.append(f"source-file analysis lacks nesting claim boundary: {path}")
    benchmark = Path(str(payload.get("benchmark_summary", "")))
    if not benchmark.is_file() or file_sha256(benchmark) != payload.get("benchmark_summary_sha256"):
        errors.append(f"source-file benchmark provenance mismatch: {path}")
    files = payload.get("files", {})
    csv_path = Path(str(files.get("source_rows_csv", "")))
    markdown_path = Path(str(files.get("summary_markdown", "")))
    for artifact, expected_hash in (
        (csv_path, files.get("source_rows_csv_sha256")),
        (markdown_path, files.get("summary_markdown_sha256")),
    ):
        if not artifact.is_file() or file_sha256(artifact) != expected_hash:
            errors.append(f"source-file report artifact mismatch: {artifact}")
    expected_rows = 0
    for seed, methods in payload.get("aggregate", {}).items():
        for method, item in methods.items():
            source_count = int(item.get("source_file_count", 0))
            expected_rows += source_count
            interval = item.get("macro_source_window_accuracy_bootstrap_95ci", [])
            if source_count <= 0 or len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1:
                errors.append(f"invalid source aggregate for seed={seed}, method={method}: {path}")
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            observed_rows = len(rows)
        if observed_rows != expected_rows:
            errors.append(f"source row count mismatch {observed_rows}/{expected_rows}: {path}")
        unique = {(row["seed"], row["method"], row["source_file"]) for row in rows}
        if len(unique) != len(rows):
            errors.append(f"duplicate source CSV identities: {path}")
        for seed, methods in payload.get("aggregate", {}).items():
            for method, item in methods.items():
                selected = [row for row in rows if row["seed"] == str(seed) and row["method"] == method]
                if len(selected) != item["source_file_count"]:
                    errors.append(f"source CSV cell coverage mismatch: {seed}/{method}")
                    continue
                names = sorted(item["majority_vote_per_class_recall"])
                mapping = {name: index for index, name in enumerate(names)}
                matrix = np.zeros((len(names), len(names)), dtype=np.int64)
                failed = []
                for row in selected:
                    truth = mapping[row["true_label"]]
                    pred = mapping[row["majority_predicted_label"]]
                    matrix[truth, pred] += 1
                    if (truth == pred) != (row["majority_correct"] == "True"):
                        errors.append("source CSV correctness flag differs")
                    count, correct = int(row["window_count"]), int(row["correct_window_count"])
                    if count <= 0 or not 0 <= correct <= count or not np.isclose(float(row["window_accuracy"]), correct / count, atol=1e-10, rtol=0):
                        errors.append("source CSV window accuracy differs")
                    if truth != pred:
                        failed.append(row["source_label"])
                metric = {"confusion_matrix": matrix.tolist(), "sample_count": len(selected),
                          "macro_f1": item["majority_vote_macro_f1"], "accuracy": item["majority_vote_accuracy"],
                          "per_class_recall": item["majority_vote_per_class_recall"]}
                errors.extend(confusion_errors(metric, names))
                if sorted(failed) != sorted(item["failed_source_files"]):
                    errors.append(f"failed source identities differ: {seed}/{method}")
    for seed, methods in payload.get("run_provenance", {}).items():
        for method, item in methods.items():
            for path_key, hash_key in (
                ("manifest", "manifest_sha256"),
                ("prediction_file", "prediction_sha256"),
            ):
                artifact = Path(str(item.get(path_key, "")))
                if not artifact.is_file() or file_sha256(artifact) != item.get(hash_key):
                    errors.append(f"source provenance mismatch for {seed}/{method}: {artifact}")
    return errors


def validate_cwru_extension(path: Path) -> list[str]:
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_methods = {"no_cs", "ilsomp", "downsample", "pca", "random_projection"}
    expected_seeds = {"42", "100", "200", "300", "400"}
    if not payload.get("complete") or payload.get("status") != "complete":
        errors.append(f"incomplete CWRU five-seed extension: {path}")
    if payload.get("stage") != "external-cwru-five-seed-extension":
        errors.append(f"unexpected CWRU extension stage: {path}")
    if not payload.get("paper_claim_authorized") or not payload.get("claim_boundary"):
        errors.append(f"CWRU extension lacks disclosure/claim authorization: {path}")
    base = Path(str(payload.get("base_summary", "")))
    if not base.is_file() or file_sha256(base) != payload.get("base_summary_sha256"):
        errors.append(f"CWRU extension base summary changed: {path}")
    runs = payload.get("runs", {})
    if set(runs) != expected_seeds:
        errors.append(f"CWRU extension seed set mismatch: {path}")
    for seed in expected_seeds:
        if set(runs.get(seed, {})) != expected_methods:
            errors.append(f"CWRU extension method set mismatch for seed {seed}: {path}")
    if set(payload.get("extension_runs", {})) != {"300", "400"}:
        errors.append(f"CWRU extension does not isolate seeds 300/400: {path}")
    for seed, methods in payload.get("extension_runs", {}).items():
        for method, run in methods.items():
            for path_key, hash_key in (
                ("manifest_path", "manifest_sha256"),
                ("metrics_path", "metrics_sha256"),
            ):
                artifact = Path(str(run.get(path_key, "")))
                if not artifact.is_file() or file_sha256(artifact) != run.get(hash_key):
                    errors.append(
                        f"CWRU extension provenance mismatch for {seed}/{method}: {artifact}"
                    )
    for method, item in payload.get("aggregate", {}).get("methods", {}).items():
        if method in expected_methods and item.get("n") != 5:
            errors.append(f"CWRU extension aggregate {method} lacks n=5: {path}")
        selected = [runs.get(seed, {}).get(method, {}) for seed in sorted(expected_seeds)]
        if any("paper_metrics" not in run for run in selected):
            errors.append(f"missing CWRU metrics for {method}: {path}")
            continue
        for run in selected:
            errors.extend(confusion_errors(run["paper_metrics"] | {"per_class_recalls": run["per_class_recalls"]}, run.get("class_names")))
        for metric in ("accuracy", "macro_f1"):
            errors.extend(statistic_errors(item.get(metric, {}), [run["paper_metrics"][metric] for run in selected]))
        errors.extend(statistic_errors(item.get("training_epoch_seconds", {}), [run["training_epoch_seconds"] for run in selected]))
        for name, saved in item.get("per_class_recall", {}).items():
            errors.extend(statistic_errors(saved, [run["per_class_recalls"][name] for run in selected]))
        errors.extend(statistic_errors(item.get("paired_macro_f1_delta_vs_no_cs", {}),
                     [runs[seed][method]["paper_metrics"]["macro_f1"] - runs[seed]["no_cs"]["paper_metrics"]["macro_f1"] for seed in sorted(expected_seeds)]))
    return errors


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "artifacts_v2" / "reports" / "source_file_performance",
    )
    cli.add_argument(
        "--study-root",
        type=Path,
        default=ROOT / "artifacts_v2" / "studies" / "benchmark",
    )
    args = cli.parse_args()
    source_summaries = list(args.source_root.glob("*_source_file_summary.json")) if args.source_root.is_dir() else []
    extension_summaries = list(args.study_root.glob("external_cwru_five_seed_*_summary.json")) if args.study_root.is_dir() else []
    errors: list[str] = []
    for path in source_summaries:
        errors.extend(validate_source_summary(path))
    for path in extension_summaries:
        errors.extend(validate_cwru_extension(path))
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        raise SystemExit(1)
    if not source_summaries and not extension_summaries:
        print("OK: reviewer-extension framework ready; no result artifacts exist yet")
    else:
        print(
            f"OK: validated {len(source_summaries)} source-file summaries and "
            f"{len(extension_summaries)} CWRU five-seed summaries"
        )


if __name__ == "__main__":
    main()
