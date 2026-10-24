





from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_clean_study import load_config, load_samples, run_condition
from scripts.run_paired_cs_study import _split_fingerprint
from scripts.validate_study_artifacts import validate_manifest, validate_metrics
from src.utils.io_tools import export_json, stratified_group_split_three_way
from src.utils.reproducibility import canonical_hash, file_sha256


DEFAULT_PROTOCOL = ROOT / "configs" / "leakage_aware_benchmark_v1.json"
FORMAL_STAGES = {"formal-pu", "leakage-audit", "external-cwru"}


def _load_protocol(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "study_name", "development_dataset", "external_dataset",
        "representation", "primary_split_unit", "leakage_control_split_unit",
        "retention_ratio", "methods", "leakage_audit_methods", "development_seeds",
        "formal_seeds", "training", "compressed_sensing", "stft", "claim_rules",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"benchmark protocol is missing fields: {sorted(missing)}")
    if payload["primary_split_unit"] != "source-file":
        raise ValueError("paper-valid benchmark must use source-file splitting")
    if payload["leakage_control_split_unit"] != "window":
        raise ValueError("leakage audit must use the explicitly invalid window control")
    if payload["methods"][0] != "no_cs" or set(payload["methods"]) != {
        "no_cs", "ilsomp", "downsample", "pca", "random_projection"
    }:
        raise ValueError("matched-budget method set is not the locked five-arm benchmark")
    return payload, canonical_hash(payload, length=24)


def _stage_design(protocol: dict, stage: str) -> dict:
    if stage == "validation":
        return {
            "dataset": protocol["development_dataset"],
            "retention_ratio": float(protocol["retention_ratio"]),
            "split_unit": protocol["primary_split_unit"],
            "seeds": list(protocol["development_seeds"]),
            "methods": list(protocol["methods"]),
            "evaluation_scope": "validation-only",
            "paper_claim_authorized": False,
        }
    if stage == "formal-pu":
        return {
            "dataset": protocol["development_dataset"],
            "retention_ratio": float(protocol["retention_ratio"]),
            "split_unit": protocol["primary_split_unit"],
            "seeds": list(protocol["formal_seeds"]),
            "methods": list(protocol["methods"]),
            "evaluation_scope": "formal-test",
            "paper_claim_authorized": True,
        }
    if stage == "leakage-audit":
        return {
            "dataset": protocol["development_dataset"],
            "retention_ratio": float(protocol["retention_ratio"]),
            "split_unit": protocol["leakage_control_split_unit"],
            "seeds": list(protocol["formal_seeds"]),
            "methods": list(protocol["leakage_audit_methods"]),
            "evaluation_scope": "formal-test",
            "paper_claim_authorized": False,
        }
    if stage == "external-cwru":
        return {
            "dataset": protocol["external_dataset"],
            "retention_ratio": float(protocol["retention_ratio"]),
            "split_unit": protocol["primary_split_unit"],
            "seeds": list(protocol["formal_seeds"]),
            "methods": list(protocol["methods"]),
            "evaluation_scope": "formal-test",
            "paper_claim_authorized": True,
        }
    raise ValueError(f"unknown benchmark stage: {stage}")


def _condition_args(protocol: dict, design: dict, method_name: str, seed: int) -> Namespace:
    training = protocol["training"]
    sensing = protocol["compressed_sensing"]
    stft = protocol["stft"]
    no_cs = method_name == "no_cs"
    compression_baseline = method_name if method_name in {"downsample", "pca", "random_projection"} else None
    return Namespace(
        config=ROOT / "configs" / "default_config.yaml",
        dataset=design["dataset"], method=sensing["method"],
        ratio=float(protocol["retention_ratio"]), representation=protocol["representation"],
        backbone=training["backbone"], seed=int(seed), epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]), batch_size=int(training["batch_size"]),
        freeze_ratio=float(training["freeze_ratio"]), num_workers=int(training["num_workers"]),
        amp=bool(training["amp"]), class_weighting=training["class_weighting"], sampler="none",
        report_train_metrics=False, checkpoint_selection=training["checkpoint_selection"],
        evaluation_scope=design["evaluation_scope"], fast_training=bool(training["fast_training"]),
        prefetch_factor=int(training["prefetch_factor"]), eval_num_workers=int(training["eval_num_workers"]),
        prepare_workers=int(training["prepare_workers"]), prepare_chunksize=int(training["prepare_chunksize"]),
        no_cs=no_cs, sparsity_k=int(sensing["sparsity_k"]),
        gradient_lambda=float(sensing["gradient_lambda"]), sparse_basis=sensing["sparse_basis"],
        diagnostic_prior_mu=0.5, diagnostic_prior_max_per_class=4096,
        wavelet=None, cwt_scales_start=1, cwt_scales_end=128,
        stft_nperseg=int(stft["nperseg"]), stft_noverlap=int(stft["noverlap"]),
        noise_snr_db=None, train_conditions=(), test_condition=None,
        compression_baseline=compression_baseline, pca_fit_samples=4096,
        test_groups="", val_groups="", split_unit=design["split_unit"],
        prepare_only=False, max_samples_per_group=None,
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "calibrating"
    minutes = int(round(seconds)) // 60
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def _metrics_file(result_root: Path, backbone: str) -> Path:
    return result_root / f"{backbone}_metrics.json"


def _per_class_recalls(metrics: dict, class_names: list[str]) -> dict[str, float]:
    matrix = np.asarray(metrics.get("confusion_matrix"), dtype=np.float64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError(
            f"confusion matrix shape {matrix.shape} does not match {len(class_names)} classes"
        )
    denominators = matrix.sum(axis=1)
    if np.any(denominators <= 0):
        raise ValueError("confusion matrix contains an empty evaluation class")
    return {
        class_name: float(value)
        for class_name, value in zip(class_names, np.diag(matrix) / denominators)
    }


def _validate_and_enrich(
    run: dict,
    protocol: dict,
    design: dict,
    method_name: str,
    seed: int,
    runner_elapsed: float,
) -> dict:
    dataset_root = Path(run["dataset_root"])
    result_root = Path(run["result_root"])
    manifest_path = dataset_root / "dataset_manifest.json"
    metrics_path = _metrics_file(result_root, protocol["training"]["backbone"])
    errors = validate_manifest(manifest_path) + validate_metrics(metrics_path)
    if errors:
        raise RuntimeError("artifact validation failed:\n" + "\n".join(errors))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    split = manifest.get("condition", {}).get("split", {})
    expected_mode = (
        "deliberately_leaky_window_control"
        if design["split_unit"] == "window"
        else "stratified_source_file_random"
    )
    if split.get("mode") != expected_mode:
        raise RuntimeError(f"unexpected split mode for {method_name}/seed{seed}: {split.get('mode')}")
    if metrics_payload.get("evaluation_scope", "formal-test") != design["evaluation_scope"]:
        raise RuntimeError(f"evaluation scope mismatch for {method_name}/seed{seed}")
    metric_key = "validation_metrics" if design["evaluation_scope"] == "validation-only" else "paper_metrics"
    metrics = run.get(metric_key)
    if not isinstance(metrics, dict):
        raise RuntimeError(f"missing {metric_key} for {method_name}/seed{seed}")
    class_names = list(metrics_payload.get("class_names", []))
    recalls = _per_class_recalls(metrics, class_names)
    history = metrics_payload.get("history", {})
    enriched = dict(run)
    enriched.update({
        "benchmark_method": method_name,
        "artifact_validation": "passed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": file_sha256(metrics_path),
        "split_fingerprint": _split_fingerprint(manifest),
        "class_names": class_names,
        "per_class_recalls": recalls,
        "runner_elapsed_seconds": float(runner_elapsed),
        "training_epoch_seconds": float(sum(history.get("epoch_seconds", []))),
        "paper_claim_authorized": bool(design["paper_claim_authorized"]),
    })
    if design["split_unit"] == "window":
        enriched["invalid_control_warning"] = (
            "Deliberately leaky window-level control; usable only to quantify leakage inflation."
        )
    return enriched


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "n": int(len(array)),
    }


def aggregate(summary: dict) -> dict:
    design = summary["stage_design"]
    metric_key = "validation_metrics" if design["evaluation_scope"] == "validation-only" else "paper_metrics"
    runs = summary.get("runs", {})
    method_stats: dict[str, dict] = {}
    baseline_by_seed: dict[str, dict] = {}
    for seed in design["seeds"]:
        baseline = runs.get(str(seed), {}).get("no_cs")
        if baseline:
            baseline_by_seed[str(seed)] = baseline
    for method_name in design["methods"]:
        selected = [
            runs[str(seed)][method_name]
            for seed in design["seeds"]
            if method_name in runs.get(str(seed), {})
        ]
        if not selected:
            continue
        item = {
            "n": len(selected),
            "accuracy": _stats([float(run[metric_key]["accuracy"]) for run in selected]),
            "macro_f1": _stats([float(run[metric_key]["macro_f1"]) for run in selected]),
            "training_epoch_seconds": _stats([float(run["training_epoch_seconds"]) for run in selected]),
            "per_class_recall": {},
        }
        class_names = selected[0]["class_names"]
        for class_name in class_names:
            item["per_class_recall"][class_name] = _stats(
                [float(run["per_class_recalls"][class_name]) for run in selected]
            )
        paired_deltas = []
        for run in selected:
            baseline = baseline_by_seed.get(str(run["seed"]))
            if baseline is not None:
                paired_deltas.append(
                    float(run[metric_key]["macro_f1"] - baseline[metric_key]["macro_f1"])
                )
        if paired_deltas:
            item["paired_macro_f1_delta_vs_no_cs"] = _stats(paired_deltas)
        method_stats[method_name] = item
    return {"metric_scope": metric_key, "methods": method_stats}


def _summary_path(protocol_signature: str, stage: str) -> Path:
    safe_stage = stage.replace("-", "_")
    return ROOT / "artifacts_v2" / "studies" / "benchmark" / f"{safe_stage}_{protocol_signature}_summary.json"


def _require_prerequisite(protocol_signature: str, stage: str) -> None:
    required_stage = "validation" if stage == "formal-pu" else "formal-pu"
    path = _summary_path(protocol_signature, required_stage)
    if not path.is_file():
        raise RuntimeError(f"{stage} requires the completed {required_stage} stage: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise RuntimeError(f"{required_stage} is incomplete: {path}")
    if required_stage == "validation" and payload.get("independent_test_opened") is not False:
        raise RuntimeError("validation prerequisite improperly opened the independent test")


def _planned_cells(design: dict) -> list[tuple[int, str]]:
    return [(int(seed), method) for seed in design["seeds"] for method in design["methods"]]


def _preflight_grouped_dataset(
    samples: list,
    train_ratio: float,
    val_ratio: float,
    seeds: list[int],
) -> dict:
    
    if not samples:
        raise RuntimeError("preflight loaded zero samples")
    labels = [str(sample.label) for sample in samples]
    groups = [str(Path(sample.source_file).resolve()) for sample in samples]
    expected_classes = sorted(set(labels))
    class_groups: dict[str, set[str]] = defaultdict(set)
    for label, group in zip(labels, groups):
        class_groups[label].add(group)
    insufficient = {
        label: len(units) for label, units in class_groups.items() if len(units) < 3
    }
    if insufficient:
        raise RuntimeError(
            "external grouped split requires at least three source files per class: "
            f"{insufficient}"
        )

    split_audits: dict[str, dict] = {}
    for seed in seeds:
        arrays = stratified_group_split_three_way(
            labels, groups, train_ratio, val_ratio, int(seed)
        )
        per_split: dict[str, dict] = {}
        split_group_lists: dict[str, list[str]] = {}
        for split_name, indices in zip(("train", "val", "test"), arrays):
            index_list = [int(index) for index in indices]
            split_labels = [labels[index] for index in index_list]
            split_groups = sorted({groups[index] for index in index_list})
            observed_classes = sorted(set(split_labels))
            if observed_classes != expected_classes:
                raise RuntimeError(
                    f"seed {seed} {split_name} lacks classes: "
                    f"{sorted(set(expected_classes) - set(observed_classes))}"
                )
            per_split[split_name] = {
                "sample_count": len(index_list),
                "source_file_count": len(split_groups),
                "class_sample_counts": dict(sorted(Counter(split_labels).items())),
                "class_source_file_counts": {
                    class_name: len({
                        groups[index]
                        for index in index_list
                        if labels[index] == class_name
                    })
                    for class_name in expected_classes
                },
            }
            split_group_lists[split_name] = split_groups
        per_split["split_fingerprint"] = canonical_hash(split_group_lists, length=24)
        split_audits[str(seed)] = per_split

    return {
        "sample_count": len(samples),
        "class_sample_counts": dict(sorted(Counter(labels).items())),
        "class_source_file_counts": {
            label: len(class_groups[label]) for label in expected_classes
        },
        "split_audits": split_audits,
        "ready": True,
        "independent_test_opened": False,
    }


def _stage_is_complete(path: Path, stage: str, protocol_signature: str) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        payload.get("complete")
        and payload.get("stage") == stage
        and payload.get("protocol_signature") == protocol_signature
    )


def _external_preflight(protocol: dict, protocol_signature: str, config_path: Path | None = None) -> dict:
    cfg = load_config(config_path or ROOT / "configs" / "default_config.yaml")
    dataset = str(protocol["external_dataset"])
    samples = load_samples(cfg, dataset)
    required_classes = {str(value) for value in cfg["signal"]["classes"]}
    observed_classes = {str(sample.label) for sample in samples}
    if observed_classes != required_classes:
        raise RuntimeError(
            "external dataset class coverage mismatch: "
            f"missing={sorted(required_classes - observed_classes)}, "
            f"unexpected={sorted(observed_classes - required_classes)}"
        )
    audit = _preflight_grouped_dataset(
        samples,
        float(cfg["training"]["train_ratio"]),
        float(cfg["training"]["val_ratio"]),
        [int(seed) for seed in protocol["formal_seeds"]],
    )
    formal_complete = _stage_is_complete(
        _summary_path(protocol_signature, "formal-pu"),
        "formal-pu",
        protocol_signature,
    )
    return {
        "status": "ready",
        "scope": "metadata-only external-dataset preflight",
        "dataset": dataset,
        "dataset_root": str(Path(cfg["paths"][f"{dataset}_root"]).resolve()),
        "methods": list(protocol["methods"]),
        "seeds": list(protocol["formal_seeds"]),
        **audit,
        "formal_pu_complete": formal_complete,
        "next_stage_blocked_until_formal_pu_complete": not formal_complete,
    }


def _audit_resumed_runs(summary: dict, protocol: dict, design: dict) -> None:
    allowed = set(_planned_cells(design))
    observed: set[tuple[int, str]] = set()
    for seed_text, methods in summary.get("runs", {}).items():
        seed = int(seed_text)
        for method_name, run in methods.items():
            cell = (seed, method_name)
            if cell not in allowed:
                raise RuntimeError(f"resumed summary contains an unexpected cell: {cell}")
            observed.add(cell)
            _validate_and_enrich(
                run,
                protocol,
                design,
                method_name,
                seed,
                float(run.get("runner_elapsed_seconds", 0.0)),
            )
    for seed in design["seeds"]:
        seed_runs = summary.get("runs", {}).get(str(seed), {})
        baseline = seed_runs.get("no_cs")
        if baseline is None:
            continue
        for method_name, run in seed_runs.items():
            if run.get("split_fingerprint") != baseline.get("split_fingerprint"):
                raise RuntimeError(
                    f"resumed split mismatch for seed={seed}: no_cs vs {method_name}"
                )


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    cli.add_argument("--config", type=Path, default=ROOT / "configs" / "default_config.yaml")
    cli.add_argument("--stage", choices=("validation", "formal-pu", "leakage-audit", "external-cwru"))
    cli.add_argument("--show-protocol", action="store_true")
    cli.add_argument(
        "--preflight-external",
        action="store_true",
        help="metadata-only external-data audit; never trains or evaluates test metrics",
    )
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--confirm-protocol-signature", default="")
    args = cli.parse_args()

    protocol, protocol_signature = _load_protocol(args.protocol)
    if args.show_protocol:
        print(json.dumps({
            "protocol_path": str(args.protocol.resolve()),
            "protocol_signature": protocol_signature,
            "protocol": protocol,
        }, indent=2, ensure_ascii=False))
        return
    if args.preflight_external:
        if args.stage:
            raise SystemExit("--preflight-external cannot be combined with --stage")
        print(json.dumps(
            _external_preflight(protocol, protocol_signature, args.config),
            indent=2,
            ensure_ascii=False,
        ))
        return
    if not args.stage:
        raise SystemExit("choose --stage or use --show-protocol")
    design = _stage_design(protocol, args.stage)
    if args.stage in FORMAL_STAGES:
        if args.confirm_protocol_signature != protocol_signature:
            raise SystemExit(
                "formal/test-bearing stage blocked: run --show-protocol, review it, then pass "
                f"--confirm-protocol-signature {protocol_signature}"
            )
        _require_prerequisite(protocol_signature, args.stage)

    output_path = _summary_path(protocol_signature, args.stage)
    stage_signature = canonical_hash({
        "protocol_signature": protocol_signature,
        "stage": args.stage,
        "design": design,
    }, length=24)
    summary = {
        "schema_version": 1,
        "complete": False,
        "status": "initializing",
        "stage": args.stage,
        "stage_signature": stage_signature,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_signature": protocol_signature,
        "protocol_sha256": file_sha256(args.protocol),
        "stage_design": design,
        "independent_test_opened": False,
        "runs": {},
    }
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("stage_signature") != stage_signature:
            raise RuntimeError(f"refusing incompatible stage summary: {output_path}")
        summary = previous
        _audit_resumed_runs(summary, protocol, design)
        print(f"resuming benchmark stage: {output_path}")

    cells = _planned_cells(design)
    completed = {
        (int(seed), method)
        for seed, values in summary.get("runs", {}).items()
        for method in values
    }
    pending = [cell for cell in cells if cell not in completed]
    if args.dry_run:
        print(json.dumps({
            "stage": args.stage,
            "protocol_signature": protocol_signature,
            "total_cells": len(cells),
            "completed_cells": len(completed),
            "pending_cells": [{"seed": seed, "method": method} for seed, method in pending],
            "independent_test_will_open": design["evaluation_scope"] == "formal-test",
            "paper_claim_authorized": design["paper_claim_authorized"],
        }, indent=2, ensure_ascii=False))
        return

    cfg = load_config(args.config)
    observed = [
        float(run.get("runner_elapsed_seconds", 0.0))
        for values in summary.get("runs", {}).values()
        for run in values.values()
        if float(run.get("runner_elapsed_seconds", 0.0)) >= 60.0
    ]
    for cell_index, (seed, method_name) in enumerate(cells, 1):
        if (seed, method_name) in completed:
            print(f"[{args.stage}] seed={seed}, method={method_name}: complete; skipping")
            continue
        mean_seconds = float(np.mean(observed)) if observed else None
        remaining_after = len(cells) - cell_index
        print(
            f"\n[{args.stage} {cell_index}/{len(cells)}] seed={seed}, method={method_name}; "
            f"ETA after current={_format_duration(mean_seconds * remaining_after if mean_seconds else None)}"
        )
        started = time.perf_counter()
        raw_run = run_condition(_condition_args(protocol, design, method_name, seed), cfg)
        elapsed = time.perf_counter() - started
        run = _validate_and_enrich(raw_run, protocol, design, method_name, seed, elapsed)
        baseline = summary["runs"].get(str(seed), {}).get("no_cs")
        if baseline is not None and run["split_fingerprint"] != baseline["split_fingerprint"]:
            raise RuntimeError(f"split mismatch for seed={seed}: no_cs vs {method_name}")
        summary["runs"].setdefault(str(seed), {})[method_name] = run
        summary["aggregate"] = aggregate(summary)
        summary["status"] = "running"
        summary["independent_test_opened"] = bool(
            design["evaluation_scope"] == "formal-test" and summary["runs"]
        )
        export_json(output_path, summary)
        if elapsed >= 60.0:
            observed.append(elapsed)

    completed = {
        (int(seed), method)
        for seed, values in summary.get("runs", {}).items()
        for method in values
    }
    summary["aggregate"] = aggregate(summary)
    summary["complete"] = completed == set(cells)
    summary["status"] = "complete" if summary["complete"] else "incomplete"
    summary["independent_test_opened"] = design["evaluation_scope"] == "formal-test" and bool(completed)
    summary["paper_claim_authorized"] = bool(design["paper_claim_authorized"] and summary["complete"])
    if args.stage == "leakage-audit":
        summary["claim_boundary"] = (
            "Leaky-control numbers may quantify inflation only; they are not valid generalization performance."
        )
    export_json(output_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "stage": args.stage,
        "protocol_signature": protocol_signature,
        "summary": str(output_path),
        "aggregate": summary["aggregate"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
