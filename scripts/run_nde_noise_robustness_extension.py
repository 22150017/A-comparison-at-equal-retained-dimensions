from __future__ import annotations

import argparse
import json
from pathlib import Path

from rie_supplement_support import METHODS, aggregate, align, arrays, artifact, build_protocol, evaluate, load_lock, ordered_records, read_json, save_json, signature, verify_inputs
from run_rie_noise_sensitivity import run as run_noise


DEFAULT_REUSE = Path("F:/paper1/artifacts_v2/studies/rie_noise_sensitivity/4f52371b6d86d971763e57d4/protocol_lock.json")
DEFAULT_ROOT = Path("F:/paper1/artifacts_v2/studies/nde_noise_robustness_extension")
CONFIG_KEYS = ("methods", "retained_values", "window_samples", "sampling_rate_hz", "snr_db", "noise_master_seed", "noise_rule", "noise_power_reference", "pca_refitted", "checkpoint_selection_changed", "source_vote_rule", "inference_precision", "batch_size", "chunk_size")


def build_master(reused_lock: Path) -> dict:
    reused = load_lock(reused_lock.resolve(), "noise_sensitivity")
    old = reused["protocol"]
    if [(r["dataset"], r["seed"]) for r in old["runs"]] != [("pu", 42)]:
        raise ValueError("Reuse is restricted to original PU seed42")
    old_summary_path = reused_lock.resolve().parent / "noise_sensitivity_summary.json"
    old_summary = read_json(old_summary_path)
    if not old_summary["complete"] or old_summary["signature"] != reused["signature"] or len(old_summary["cells"]) != 10:
        raise ValueError("Original ten-cell result is not complete")
    jobs = []
    inputs = [artifact(reused_lock), artifact(old_summary_path)]
    inputs.extend(old["inputs"])
    for dataset, seeds in (("pu", [100, 200]), ("cwru", [42, 100, 200, 300, 400])):
        protocol = build_protocol("noise_sensitivity", [dataset], seeds, 32, 256)
        if any(protocol[key] != old[key] for key in CONFIG_KEYS):
            raise ValueError("Extension differs from original noise procedure")
        child_signature = signature(protocol)
        jobs.append({"dataset": dataset, "seeds": seeds, "signature": child_signature,
                     "relative_lock": f"jobs/{child_signature}/protocol_lock.json", "protocol": protocol})
        inputs.extend(protocol["inputs"])
    inputs.extend(artifact(Path(__file__).with_name(name)) for name in
                  ("run_nde_noise_robustness_extension.py", "validate_nde_noise_robustness_extension.py"))
    unique = {}
    for item in inputs:
        if item["path"] in unique and unique[item["path"]] != item["sha256"]:
            raise ValueError("Conflicting input hashes")
        unique[item["path"]] = item["sha256"]
    return {"schema_version": 1, "experiment": "nde_noise_robustness_extension",
            "reused_lock": artifact(reused_lock), "reused_summary": artifact(old_summary_path),
            "jobs": jobs, "inputs": [{"path": path, "sha256": sha} for path, sha in sorted(unique.items())],
            "design": {"datasets_and_seeds": {"pu": [42, 100, 200], "cwru": [42, 100, 200, 300, 400]},
                       "methods": METHODS, "snr_db": [20.0, 10.0], "noise_master_seed": 20260930,
                       "retained_values": 512, "window_samples": 1024, "sampling_rate_hz": 12000,
                       "primary_contrast": "PCA minus uncompressed window Macro-F1 at original/20dB/10dB; paired noise-minus-original contrast change",
                       "secondary_metrics": ["source_macro_f1", "minimum_class_recall", "failed_source_count", "per_class_recall"],
                       "reversal_definition": "original paired difference < 0 and noisy paired difference > 0",
                       "summary_statistics": "per-run values, mean, sample SD, minimum, maximum and sign counts",
                       "inference_policy": "descriptive sensitivity analysis; overlapping source identities and 3/5 runs are not independent physical replications; no population p-values or confidence claims",
                       "all_locked_cells_required": True, "outcome_dependent_continuation": False,
                       "checkpoint_selection_changed": False, "pca_refitted": False},
            "total_noise_cells": 80, "new_inference_cells": 70, "reused_prediction_cells": 10,
            "training_performed": False, "new_independent_test_opened": False, "uored_test_opened": False}


def load_master(path: Path) -> dict:
    lock = read_json(path)
    protocol = lock["protocol"]
    if protocol["experiment"] != "nde_noise_robustness_extension" or lock["signature"] != signature(protocol):
        raise ValueError("Invalid master protocol")
    verify_inputs(protocol)
    return lock


def prepare_master(args: argparse.Namespace) -> None:
    protocol = build_master(args.reuse_lock.resolve())
    value = signature(protocol)
    root = args.output_root.resolve() / value
    preview = {"signature": value, "lock": str(root / "protocol_lock.json"), "training_cells": 0,
               "total_noise_cells": 80, "new_inference_cells": 70, "reused_prediction_cells": 10,
               "jobs": [{"dataset": j["dataset"], "seeds": j["seeds"], "inference_cells": j["protocol"]["inference_cells"]} for j in protocol["jobs"]],
               "snr_db": [20.0, 10.0], "noise_master_seed": 20260930,
               "independent_confirmation": False, "uored_test_opened": False}
    if not args.dry_run:
        if args.confirm_signature != value:
            raise SystemExit(f"Locking requires --confirm-signature {value}")
        locks = [(root / "protocol_lock.json", {"signature": value, "protocol": protocol})]
        locks.extend((root / job["relative_lock"], {"signature": job["signature"], "protocol": job["protocol"]}) for job in protocol["jobs"])
        for path, content in locks:
            if path.exists() and read_json(path) != content:
                raise ValueError(f"Existing protocol differs: {path}")
        for path, content in locks:
            if not path.exists():
                save_json(path, content)
    print(json.dumps(preview, ensure_ascii=False, indent=2))


def run_master(path: Path) -> None:
    from validate_rie_supplement import verify
    lock = load_master(path)
    protocol, root = lock["protocol"], path.resolve().parent
    old_lock_path = Path(protocol["reused_lock"]["path"])
    verify(old_lock_path)
    sources = [{"role": "reused", "protocol_lock": artifact(old_lock_path), "summary": protocol["reused_summary"]}]
    save_json(root / "progress.json", {"signature": lock["signature"], "complete": False, "new_jobs_completed": 0,
                                      "new_inference_cells": 70, "reused_prediction_cells": 10, "training_performed": False})
    for index, job in enumerate(protocol["jobs"], 1):
        child_path = root / job["relative_lock"]
        if read_json(child_path) != {"signature": job["signature"], "protocol": job["protocol"]}:
            raise ValueError("Child protocol differs from locked design")
        print(f"[extension] starting {job['dataset']} seeds {job['seeds']}", flush=True)
        run_noise(child_path)
        verify(child_path)
        sources.append({"role": "new", "protocol_lock": artifact(child_path), "summary": artifact(child_path.parent / "noise_sensitivity_summary.json")})
        save_json(root / "progress.json", {"signature": lock["signature"], "complete": False, "new_jobs_completed": index,
                                          "new_inference_cells": 70, "reused_prediction_cells": 10, "training_performed": False})
    noise_cells, clean_cells, seen = [], [], set()
    for source in sources:
        child = read_json(source["protocol_lock"]["path"])
        summary = read_json(source["summary"]["path"])
        noise_cells.extend(summary["cells"])
        for run in child["protocol"]["runs"]:
            baseline = arrays(run["methods"]["no_cs"]["prediction"]["path"])
            for method in METHODS:
                entry = run["methods"][method]
                payload = arrays(entry["prediction"]["path"])
                align(baseline, payload)
                records = ordered_records(read_json(entry["manifest"]["path"]), payload)
                identity = (run["dataset"], run["seed"], method)
                if identity in seen:
                    raise ValueError("Duplicated clean cell")
                seen.add(identity)
                clean_cells.append({"dataset": run["dataset"], "seed": run["seed"], "method": method,
                                    "prediction": entry["prediction"], "metrics": evaluate(payload, records)})
    if len(noise_cells) != 80 or len(clean_cells) != 40:
        raise ValueError("Extension cell count differs")
    summary = {"complete": True, "signature": lock["signature"], "protocol_lock": artifact(path),
               "summary_sources": sources, "cells": noise_cells, "clean_cells": clean_cells,
               "aggregate": aggregate(noise_cells, ["dataset", "method", "snr_db"]),
               "clean_aggregate": aggregate(clean_cells, ["dataset", "method"]),
               "total_noise_cells": 80, "new_inference_cells": 70, "reused_prediction_cells": 10,
               "training_performed": False, "pca_refitted": False, "checkpoint_selection_changed": False,
               "new_independent_test_opened": False, "independent_confirmation": False, "uored_test_opened": False}
    save_json(root / "noise_robustness_extension.json", summary)
    save_json(root / "progress.json", {"signature": lock["signature"], "complete": True, "total_noise_cells": 80,
                                      "new_inference_cells": 70, "reused_prediction_cells": 10, "training_performed": False})
    print(json.dumps({"complete": True, "summary": str(root / "noise_robustness_extension.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["prepare", "run"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-signature")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--reuse-lock", type=Path, default=DEFAULT_REUSE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_master(args)
    else:
        if args.lock is None:
            parser.error("run requires --lock")
        run_master(args.lock.resolve())


if __name__ == "__main__":
    main()
