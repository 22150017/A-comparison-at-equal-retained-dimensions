






from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_clean_study import _sample_identity, load_config, load_samples
from src.reconstruction.sparse_recovery import get_reconstructors, set_cs_seed, set_gradient_lambda, set_sparse_basis
from src.utils.io_tools import export_json
from src.utils.reproducibility import canonical_hash, file_sha256


def _load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("sample_id") or row["sample_id"] in rows or not np.isfinite(float(row["snr_db"])):
                raise ValueError(f"invalid or duplicate sidecar row: {path}")
            rows[row["sample_id"]] = row
    return rows


def build(curve_path: Path, output_root: Path, config_path: Path | None = None) -> dict:
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    if not curve.get("complete") or curve.get("stage") != "formal" or not curve.get("paper_claim_authorized"):
        raise ValueError("complete paper-authorized formal retention curve is required")
    cfg = load_config(config_path or ROOT / "configs" / "default_config.yaml")
    sample_by_id = None
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for seed, methods in curve["runs"].items():
        for ratio, run in methods.get("ilsomp", {}).items():
            manifest_path = Path(run["manifest_path"]); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("complete") or file_sha256(manifest_path) != run["manifest_sha256"]:
                raise ValueError("changed or incomplete reconstruction manifest")
            test_records = [record for record in manifest["records"] if record.get("split") == "test"]
            if len({r["sample_id"] for r in test_records}) != len(test_records):
                raise ValueError("duplicate test sample IDs")
            key = f"seed{seed}_r{ratio.replace('.', '')}"
            signature = canonical_hash({"manifest_sha256": file_sha256(manifest_path), "seed": int(seed), "ratio": float(ratio), "method": "ilsomp", "k": 64, "basis": "dct", "lambda": 0.25}, length=24)
            jsonl = output_root / f"{key}_{signature}.jsonl"; complete_path = output_root / f"{key}_{signature}_summary.json"
            completed = _load_jsonl(jsonl)
            if not set(completed).issubset({record["sample_id"] for record in test_records}):
                raise ValueError("sidecar contains unexpected test sample IDs")
            set_cs_seed(int(seed)); set_sparse_basis("dct"); set_gradient_lambda(0.25); reconstruct = get_reconstructors()["ilsomp"]
            needs_raw = any(record["sample_id"] not in completed and record.get("snr_db") is None for record in test_records)
            if needs_raw:
                if cfg["signal"] != manifest["condition"]["signal"]:
                    raise ValueError("local preprocessing differs from locked manifest")
                for source in manifest["condition"]["sources"]:
                    if file_sha256(Path(source["path"])) != source["sha256"]:
                        raise ValueError(f"raw source fingerprint differs: {source['path']}")
                if sample_by_id is None:
                    samples = load_samples(cfg, curve["design"]["dataset"])
                    sample_by_id = {_sample_identity(sample): sample for sample in samples}
            handle = jsonl.open("a", encoding="utf-8", buffering=1)
            try:
                for index, record in enumerate(test_records, 1):
                    sample_id = record["sample_id"]
                    if sample_id in completed: continue
                    if record.get("snr_db") is not None:
                        row = {"sample_id": sample_id, "snr_db": float(record["snr_db"]), "source": "manifest"}
                    else:
                        sample = sample_by_id.get(sample_id)
                        if sample is None: raise RuntimeError(f"raw sample not found for {sample_id}")
                        result = reconstruct(sample.signal, float(ratio), 64)
                        row = {"sample_id": sample_id, "snr_db": float(result.snr_db), "reconstruction_error": float(result.reconstruction_error), "source": "recomputed_sidecar"}
                    if not np.isfinite(row["snr_db"]):
                        raise ValueError("nonfinite reconstruction SNR")
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n"); completed[sample_id] = row
                    if index % 100 == 0: handle.flush(); os.fsync(handle.fileno()); print(f"[quality {key}] {len(completed)}/{len(test_records)}")
            finally:
                handle.close()
            if set(completed) != {record["sample_id"] for record in test_records}:
                raise RuntimeError(f"incomplete quality sidecar: {key}")
            values = [float(completed[record["sample_id"]]["snr_db"]) for record in test_records]
            summary = {"complete": True, "signature": signature, "seed": int(seed), "retention_ratio": float(ratio), "sample_count": len(values), "snr_db_mean": sum(values) / len(values), "records_jsonl": str(jsonl), "records_sha256": file_sha256(jsonl), "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path)}
            export_json(complete_path, summary); summaries[key] = summary
    expected = len(curve["design"]["seeds"]) * len(curve["design"]["ratios"])
    if len(summaries) != expected:
        raise RuntimeError(f"incomplete ILSOMP quality cells: {len(summaries)}/{expected}")
    master = {"complete": True, "source_curve": str(curve_path.resolve()), "source_curve_sha256": file_sha256(curve_path), "independent_test_predictions_opened": False, "summaries": summaries}
    master_path = output_root / "reconstruction_quality_sidecars.json"; export_json(master_path, master); return master


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__); cli.add_argument("--curve-summary", type=Path, required=True); cli.add_argument("--output-root", type=Path, default=ROOT / "artifacts_v2" / "studies" / "followup" / "reconstruction_quality")
    cli.add_argument("--config", type=Path, default=ROOT / "configs" / "default_config.yaml")
    args = cli.parse_args(); result = build(args.curve_summary, args.output_root, args.config); print(json.dumps({"complete": result["complete"], "cells": len(result["summaries"])}, indent=2))


if __name__ == "__main__": main()
