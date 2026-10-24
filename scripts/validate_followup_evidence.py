
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import file_sha256
from scripts.validation_math import confusion_errors, statistic_errors


def validate_curve(path: Path) -> list[str]:
    errors = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    design = payload.get("design", {})
    if payload.get("complete") is not True or payload.get("status") != "complete":
        errors.append(f"{path}: curve is incomplete")
    seeds = design.get("seeds", [])
    methods = design.get("methods", [])
    ratios = design.get("ratios", [])
    runs = payload.get("runs", {})
    expected_ratios = {f"{ratio:.2f}" for ratio in ratios}
    if not seeds or not methods or not ratios or set(runs) != {str(seed) for seed in seeds}:
        errors.append(f"{path}: curve seed coverage mismatch")
    for seed in seeds:
        seed_runs = runs.get(str(seed), {})
        if set(seed_runs) != set(methods):
            errors.append(f"{path}: seed {seed} method coverage mismatch")
        for method in methods:
            if set(seed_runs.get(method, {})) != expected_ratios:
                errors.append(f"{path}: seed {seed} method {method} ratio coverage mismatch")
    stage = payload.get("stage")
    expected_n = 3 if stage == "formal" else 1 if stage == "development" else None
    if expected_n is None or len(seeds) != expected_n:
        errors.append(f"{path}: invalid curve stage or seed count")
    aggregate = payload.get("aggregate", {}).get("rows", {})
    expected_keys = {f"{method}@{ratio:.2f}" for method in methods for ratio in ratios}
    if set(aggregate) != expected_keys:
        errors.append(f"{path}: aggregate coverage mismatch")
    for key, row in aggregate.items():
        if row.get("n") != expected_n:
            errors.append(f"{path}: {key} aggregate seed count mismatch")
        method, ratio = key.split("@")
        selected = [runs.get(str(seed), {}).get(method, {}).get(ratio, {}) for seed in seeds]
        metric_key = "paper_metrics" if stage == "formal" else "validation_metrics"
        if any(metric_key not in run for run in selected):
            errors.append(f"{path}: missing cell metrics for {key}")
            continue
        for run in selected:
            errors.extend(confusion_errors(run[metric_key] | {"per_class_recalls": run["per_class_recalls"]}, run.get("class_names")))
        for metric in ("accuracy", "macro_f1"):
            errors.extend(statistic_errors(row.get(metric, {}), [run[metric_key][metric] for run in selected]))
        for name, saved in row.get("per_class_recall", {}).items():
            errors.extend(statistic_errors(saved, [run["per_class_recalls"][name] for run in selected]))
    if stage == "development" and payload.get("independent_test_opened") is not False:
        errors.append(f"{path}: development curve opened test")
    if payload.get("paper_claim_authorized") is not (stage == "formal"):
        errors.append(f"{path}: paper claim authorization mismatch")
    return errors


def validate_efficiency(path: Path) -> list[str]:
    errors = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True or payload.get("status") != "complete":
        errors.append(f"{path}: efficiency is incomplete")
    if payload.get("independent_test_opened") is not False:
        errors.append(f"{path}: efficiency must be training-only")
    methods = payload.get("protocol", {}).get("methods", [])
    if not methods or set(payload.get("results", {})) != set(methods):
        errors.append(f"{path}: incomplete efficiency methods")
    return errors


def validate_sidecars(path: Path) -> list[str]:
    errors = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        errors.append(f"{path}: sidecar set is incomplete")
    if payload.get("independent_test_predictions_opened") is not False:
        errors.append(f"{path}: sidecars opened independent test predictions")
    source = Path(payload["source_curve"])
    if not source.is_file() or file_sha256(source) != payload.get("source_curve_sha256"):
        errors.append(f"{path}: source curve hash mismatch")
    summaries = payload.get("summaries", {})
    if len(summaries) != 12:
        errors.append(f"{path}: expected 12 reconstruction sidecars")
    for key, item in summaries.items():
        if item.get("complete") is not True:
            errors.append(f"{path}: incomplete sidecar {key}")
        records = Path(item["records_jsonl"])
        if not records.is_file() or file_sha256(records) != item.get("records_sha256"): errors.append(f"{path}: invalid sidecar {key}")
        elif len(records.read_text(encoding="utf-8").splitlines()) != item.get("sample_count"): errors.append(f"{path}: sidecar count mismatch {key}")
    return errors


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__); cli.add_argument("--root", type=Path, default=ROOT / "artifacts_v2" / "studies" / "followup")
    args = cli.parse_args(); errors = []; counts = {"curves": 0, "efficiency": 0, "sidecars": 0}
    if args.root.is_dir():
        for path in args.root.glob("retention_*_summary.json"): errors.extend(validate_curve(path)); counts["curves"] += 1
        for path in args.root.glob("efficiency_*_summary.json"): errors.extend(validate_efficiency(path)); counts["efficiency"] += 1
        quality = args.root / "reconstruction_quality" / "reconstruction_quality_sidecars.json"
        if quality.is_file(): errors.extend(validate_sidecars(quality)); counts["sidecars"] += 1
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors)); raise SystemExit(1)
    if sum(counts.values()) == 0:
        print("OK: framework-only state; no follow-up result artifacts exist yet")
    else:
        print(f"OK: validated {counts['curves']} follow-up curves, {counts['efficiency']} efficiency summaries, and {counts['sidecars']} reconstruction-quality sidecar sets")


if __name__ == "__main__": main()
