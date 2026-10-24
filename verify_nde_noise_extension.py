from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def read(path: Path) -> dict:
    content = path.read_bytes()
    if path.suffix == ".gz":
        content = gzip.decompress(content)
    return json.loads(content)


def normalized(path: str) -> str:
    return str(path).replace("\\", "/").lower()


def independent_statistics(values: list[float]) -> dict:
    n = len(values)
    average = sum(values) / n
    ordered = sorted(values)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"values": values, "n": n, "mean": average,
            "sample_sd": (sum((x - average) ** 2 for x in values) / (n - 1)) ** 0.5,
            "median": median, "minimum": min(values), "maximum": max(values),
            "positive_runs": sum(x > 0 for x in values), "negative_runs": sum(x < 0 for x in values),
            "zero_runs": sum(x == 0 for x in values), "nonnegative_runs": sum(x >= 0 for x in values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--hashes", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()
    if not (args.hashes or args.metrics):
        parser.error("Select --hashes, --metrics, or both")
    root = args.package_root.resolve()
    sys.path.insert(0, str(root / "scripts"))
    from rie_supplement_support import arrays, signature
    from validate_rie_supplement import identical, independent_aggregate, independent_metrics, pair_check
    index = read(root / "evidence/nde_noise_extension_index.json")
    artifacts = {}
    for name in ("artifact_index.json", "rie_supplement_index.json", "nde_noise_extension_index.json"):
        content = read(root / "evidence" / name)
        for item in content["artifacts"]:
            key = normalized(item["original_path"])
            if key in artifacts and artifacts[key]["original_sha256"] != item["original_sha256"]:
                raise ValueError("Conflicting archived path fingerprints")
            artifacts[key] = item
    checked = set()

    def packaged(item: dict) -> Path:
        path = (root / "evidence" / item["package_path"]).resolve()
        if not path.is_relative_to(root / "evidence"):
            raise ValueError("Archived path leaves evidence directory")
        if path not in checked:
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != item["stored_sha256"]:
                raise ValueError("Stored evidence hash mismatch")
            decoded = gzip.decompress(payload) if item["gzip"] else payload
            if hashlib.sha256(decoded).hexdigest() != item["original_sha256"]:
                raise ValueError("Original evidence hash mismatch")
            checked.add(path)
        return path

    def resolve(item: dict) -> Path:
        stored = artifacts[normalized(item["path"])]
        if stored["original_sha256"] != item["sha256"]:
            raise ValueError("Protocol and evidence fingerprints differ")
        return packaged(stored)

    if args.hashes:
        for item in index["artifacts"]:
            packaged(item)
        for relative, expected in read(root / "NDE_EXTENSION_SHA256.json").items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Packaged implementation or index hash mismatch")
    lock = read(resolve(index["protocol_lock"]))
    summary = read(resolve(index["summary"]))
    if signature(lock["protocol"]) != lock["signature"] or summary["signature"] != lock["signature"] or not summary["complete"]:
        raise ValueError("Incomplete or mismatched master archive")
    if lock["signature"] != index["signature"]:
        raise ValueError("Index points to a different master protocol")
    for field in ("training_performed", "pca_refitted", "checkpoint_selection_changed", "new_independent_test_opened", "independent_confirmation", "uored_test_opened"):
        if summary[field]:
            raise ValueError("Scope flag differs")
    if len(summary["cells"]) != 80 or len(summary["clean_cells"]) != 40:
        raise ValueError("Archive cell coverage differs")
    noise_lookup = {(c["dataset"], c["seed"], c["method"], c["snr_db"]): c for c in summary["cells"]}
    if len(noise_lookup) != 80:
        raise ValueError("Duplicated noise cell")
    rebuilt_noise, rebuilt_clean, metrics_lookup, noises = [], [], {}, {}
    if args.metrics:
        for source in summary["summary_sources"]:
            child = read(resolve(source["protocol_lock"]))
            child_summary = read(resolve(source["summary"]))
            if signature(child["protocol"]) != child["signature"] or child_summary["signature"] != child["signature"]:
                raise ValueError("Child signature differs")
            for run in child["protocol"]["runs"]:
                baseline = arrays(resolve(run["methods"]["no_cs"]["prediction"]))
                for method in child["protocol"]["methods"]:
                    entry = run["methods"][method]
                    clean = arrays(resolve(entry["prediction"]))
                    pair_check(baseline, clean)
                    manifest = read(resolve(entry["manifest"]))
                    reference = independent_metrics(clean, manifest, "full")
                    rebuilt_clean.append({"dataset": run["dataset"], "seed": run["seed"], "method": method,
                                          "prediction": entry["prediction"], "metrics": reference})
                    metrics_lookup[(run["dataset"], run["seed"], method, "original")] = reference
                    for setting in child["protocol"]["snr_db"]:
                        stored = noise_lookup[(run["dataset"], run["seed"], method, setting)]
                        if stored["signature"] != child["signature"] or stored["training_performed"] or stored["pca_refitted"] or stored["uored_test_opened"]:
                            raise ValueError("Child cell identity or scope differs")
                        identical(stored["checkpoint"], entry["checkpoint"])
                        data = arrays(resolve(stored["prediction"]))
                        pair_check(clean, data)
                        if data["actual_added_snr_db"].shape != clean["y_true"].shape or not np.allclose(data["actual_added_snr_db"], setting, atol=1e-10, rtol=0):
                            raise ValueError("Stored noise powers differ")
                        hashes = data["noisy_input_sha256"]
                        if hashes.shape != clean["y_true"].shape or any(len(str(h)) != 64 for h in hashes):
                            raise ValueError("Invalid paired noisy input fingerprints")
                        key = (run["dataset"], run["seed"], setting)
                        if key in noises and not np.array_equal(noises[key], hashes):
                            raise ValueError("Methods did not share noisy inputs")
                        noises[key] = hashes
                        metric = independent_metrics(data, manifest, "full")
                        identical(stored["metrics"], metric)
                        identical(stored["clean_reference"], reference)
                        metrics_lookup[(run["dataset"], run["seed"], method, f"{setting:g}dB")] = metric
                        rebuilt_noise.append({"dataset": run["dataset"], "seed": run["seed"], "method": method, "snr_db": setting, "metrics": metric})
        if len(rebuilt_noise) != 80 or len(rebuilt_clean) != 40:
            raise ValueError("Missing or extra cells")
        identical(summary["clean_cells"], rebuilt_clean)
        identical(summary["aggregate"], independent_aggregate(rebuilt_noise, ["dataset", "method", "snr_db"]))
        identical(summary["clean_aggregate"], independent_aggregate(rebuilt_clean, ["dataset", "method"]))
        analysis = read(resolve(index["analysis"]))
        if len(analysis["rows"]) != 120 or len(analysis["paired_vs_uncompressed"]) != 96 or len(analysis["noise_contrast_changes"]) != 64:
            raise ValueError("Derived analysis coverage differs")
        fields = {"window_macro_f1": ("window", "macro_f1"), "source_macro_f1": ("source", "macro_f1"), "minimum_class_recall": ("window", "minimum_class_recall")}
        for row in analysis["rows"]:
            metric = metrics_lookup[(row["dataset"], row["seed"], row["method"], row["condition"])]
            for name, (level, field) in fields.items():
                identical(row[name], metric[level][field])
            identical(row["failed_source_count"], metric["failed_source_count"])
            identical(row["test_windows"], metric["window_count"])
            identical(row["test_source_files"], metric["source_file_count"])
        for row in analysis["paired_vs_uncompressed"]:
            current = metrics_lookup[(row["dataset"], row["seed"], row["method"], row["condition"])]
            base = metrics_lookup[(row["dataset"], row["seed"], "no_cs", row["condition"])]
            for name, (level, field) in fields.items():
                identical(row[name + "_delta"], current[level][field] - base[level][field])
            identical(row["failed_source_count_delta"], current["failed_source_count"] - base["failed_source_count"])
        for row in analysis["noise_contrast_changes"]:
            original = metrics_lookup[(row["dataset"], row["seed"], row["method"], "original")]
            original_ref = metrics_lookup[(row["dataset"], row["seed"], "no_cs", "original")]
            current = metrics_lookup[(row["dataset"], row["seed"], row["method"], row["condition"])]
            current_ref = metrics_lookup[(row["dataset"], row["seed"], "no_cs", row["condition"])]
            for name, (level, field) in fields.items():
                a = original[level][field] - original_ref[level][field]
                b = current[level][field] - current_ref[level][field]
                identical(row[name + "_original_delta"], a)
                identical(row[name + "_noisy_delta"], b)
                identical(row[name + "_contrast_change"], b - a)
                identical(row[name + "_strict_negative_to_positive"], a < 0 and b > 0)
                identical(row[name + "_strict_positive_to_negative"], a > 0 and b < 0)
        for row in analysis["aggregates"]:
            group = [r for r in analysis["rows"] if (r["dataset"], r["method"], r["condition"]) == (row["dataset"], row["method"], row["condition"])]
            group.sort(key=lambda r: r["seed"])
            for name in (*fields, "failed_source_count"):
                identical(row["metrics"][name], independent_statistics([r[name] for r in group]))
        for row in analysis["paired_summary"]:
            group = [r for r in analysis["paired_vs_uncompressed"] if (r["dataset"], r["method"], r["condition"]) == (row["dataset"], row["method"], row["condition"])]
            group.sort(key=lambda r: r["seed"])
            identical(row["seeds"], [r["seed"] for r in group])
            for name in (*fields, "failed_source_count"):
                identical(row["metrics"][name], independent_statistics([r[name + "_delta"] for r in group]))
            if row["condition"] != "original":
                group = [r for r in analysis["noise_contrast_changes"] if (r["dataset"], r["method"], r["condition"]) == (row["dataset"], row["method"], row["condition"])]
                group.sort(key=lambda r: r["seed"])
                for name in (*fields, "failed_source_count"):
                    identical(row["contrast_changes"][name], independent_statistics([r[name + "_contrast_change"] for r in group]))
                identical(row["strict_window_reversal_runs"], sum(r["window_macro_f1_strict_negative_to_positive"] for r in group))
    print(json.dumps({"valid": True, "signature": lock["signature"], "packaged_files_checked": len(checked),
                      "noise_cells_recomputed": len(rebuilt_noise), "clean_cells_recomputed": len(rebuilt_clean),
                      "training_performed": False, "inference_performed": False, "raw_signals_opened": False,
                      "model_weights_loaded": False, "independent_confirmation": False}, indent=2))


if __name__ == "__main__":
    main()
