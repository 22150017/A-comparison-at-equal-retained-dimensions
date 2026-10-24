from __future__ import annotations

import argparse
import json
from pathlib import Path

from rie_supplement_support import METHODS, arrays, artifact, read_json, signature, verify_inputs
from validate_rie_supplement import identical, independent_aggregate, independent_metrics, pair_check, verify


def validate(path: Path) -> dict:
    lock = read_json(path)
    protocol, root = lock["protocol"], path.resolve().parent
    if lock["signature"] != signature(protocol) or protocol["experiment"] != "nde_noise_robustness_extension":
        raise ValueError("Master protocol signature differs")
    verify_inputs(protocol)
    expected_runs = {"pu": [42, 100, 200], "cwru": [42, 100, 200, 300, 400]}
    identical(protocol["design"]["datasets_and_seeds"], expected_runs)
    identical(protocol["design"]["methods"], METHODS)
    identical(protocol["design"]["snr_db"], [20.0, 10.0])
    if protocol["training_performed"] or protocol["new_independent_test_opened"] or protocol["uored_test_opened"]:
        raise ValueError("Protocol exceeds scope")
    summary = read_json(root / "noise_robustness_extension.json")
    identical(summary["protocol_lock"], artifact(path))
    if not summary["complete"] or summary["signature"] != lock["signature"]:
        raise ValueError("Incomplete extension")
    for field in ("training_performed", "pca_refitted", "checkpoint_selection_changed", "new_independent_test_opened", "independent_confirmation", "uored_test_opened"):
        if summary[field]:
            raise ValueError(f"Scope violation: {field}")
    source_paths = [Path(protocol["reused_lock"]["path"])] + [root / job["relative_lock"] for job in protocol["jobs"]]
    if len(summary["summary_sources"]) != 3:
        raise ValueError("Unexpected source count")
    rebuilt_noise, rebuilt_clean, run_pairs = [], [], []
    for source_path, recorded in zip(source_paths, summary["summary_sources"]):
        identical(recorded["protocol_lock"], artifact(source_path))
        verify(source_path)
        child = read_json(source_path)
        child_summary_path = source_path.parent / "noise_sensitivity_summary.json"
        identical(recorded["summary"], artifact(child_summary_path))
        child_summary = read_json(child_summary_path)
        rebuilt_noise.extend(child_summary["cells"])
        for run in child["protocol"]["runs"]:
            run_pairs.append((run["dataset"], run["seed"]))
            base = arrays(run["methods"]["no_cs"]["prediction"]["path"])
            for method in METHODS:
                entry = run["methods"][method]
                data = arrays(entry["prediction"]["path"])
                pair_check(base, data)
                metrics = independent_metrics(data, read_json(entry["manifest"]["path"]), "full")
                rebuilt_clean.append({"dataset": run["dataset"], "seed": run["seed"], "method": method,
                                      "prediction": entry["prediction"], "metrics": metrics})
    expected_pairs = [(dataset, seed) for dataset, seeds in expected_runs.items() for seed in seeds]
    if sorted(run_pairs) != sorted(expected_pairs) or len(set(run_pairs)) != 8:
        raise ValueError("Dataset/seed coverage differs")
    if len(rebuilt_noise) != 80 or len(rebuilt_clean) != 40:
        raise ValueError("Cell count differs")
    identities = [(c["dataset"], c["seed"], c["method"], c["snr_db"]) for c in rebuilt_noise]
    if len(set(identities)) != 80:
        raise ValueError("Duplicate noise cell")
    identical(summary["cells"], rebuilt_noise)
    identical(summary["clean_cells"], rebuilt_clean)
    identical(summary["aggregate"], independent_aggregate(rebuilt_noise, ["dataset", "method", "snr_db"]))
    identical(summary["clean_aggregate"], independent_aggregate(rebuilt_clean, ["dataset", "method"]))
    for name, expected in (("total_noise_cells", 80), ("new_inference_cells", 70), ("reused_prediction_cells", 10)):
        identical(summary[name], expected)
    return {"valid": True, "signature": lock["signature"], "summary": str(root / "noise_robustness_extension.json"),
            "verified_noise_cells": 80, "verified_clean_prediction_cells": 40, "new_inference_cells": 70,
            "reused_prediction_cells": 10, "training_performed": False, "inference_performed_by_validator": False,
            "validator_raw_signal_values_opened": False, "uored_test_opened": False,
            "new_independent_test_opened": False, "independent_confirmation": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.lock.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
