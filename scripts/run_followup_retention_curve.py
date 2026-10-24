





from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_clean_study import load_config, run_condition
from scripts.run_leakage_aware_benchmark import (
    DEFAULT_PROTOCOL,
    _condition_args,
    _load_protocol,
    _validate_and_enrich,
)
from src.utils.io_tools import export_json
from src.utils.reproducibility import canonical_hash, file_sha256

DEFAULT_FOLLOWUP = ROOT / "configs" / "followup_evidence_v1.json"


def load_followup(path: Path) -> tuple[dict, str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base, base_signature = _load_protocol(DEFAULT_PROTOCOL)
    if payload.get("base_protocol_signature") != base_signature:
        raise ValueError("follow-up protocol does not reference the current base protocol")
    if payload.get("curve_methods") != ["ilsomp", "pca"]:
        raise ValueError("curve methods must remain locked to ILSOMP and PCA")
    ratios = [float(value) for value in payload.get("ratios", [])]
    if ratios != [0.3, 0.5, 0.7, 0.9]:
        raise ValueError("retention ratios must remain locked to 0.30/0.50/0.70/0.90")
    return payload, canonical_hash(payload, length=24), base


def curve_design(protocol: dict, stage: str) -> dict:
    if stage == "development":
        seeds = [int(protocol["development_seed"])]
        evaluation_scope = "validation-only"
    elif stage == "formal":
        seeds = [int(value) for value in protocol["seeds"]]
        evaluation_scope = "formal-test"
    else:
        raise ValueError(f"unknown curve stage: {stage}")
    return {
        "dataset": protocol["dataset"],
        "split_unit": protocol["split_unit"],
        "seeds": seeds,
        "methods": list(protocol["curve_methods"]),
        "ratios": [float(value) for value in protocol["ratios"]],
        "evaluation_scope": evaluation_scope,
        "paper_claim_authorized": stage == "formal",
    }


def aggregate_curve(summary: dict) -> dict:
    design = summary["design"]
    metric_key = "validation_metrics" if design["evaluation_scope"] == "validation-only" else "paper_metrics"
    rows: dict[str, dict] = {}
    for method in design["methods"]:
        for ratio in design["ratios"]:
            selected = [
                summary["runs"][str(seed)][method][f"{ratio:.2f}"]
                for seed in design["seeds"]
                if f"{ratio:.2f}" in summary.get("runs", {}).get(str(seed), {}).get(method, {})
            ]
            if not selected:
                continue
            values = np.asarray([run[metric_key]["macro_f1"] for run in selected], dtype=float)
            accuracies = np.asarray([run[metric_key]["accuracy"] for run in selected], dtype=float)
            rows[f"{method}@{ratio:.2f}"] = {
                "method": method,
                "retention_ratio": ratio,
                "n": len(selected),
                "macro_f1": {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None},
                "accuracy": {"mean": float(accuracies.mean()), "std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else None},
                "per_class_recall": {
                    name: {
                        "mean": float(np.mean([run["per_class_recalls"][name] for run in selected])),
                        "std": float(np.std([run["per_class_recalls"][name] for run in selected], ddof=1)) if len(selected) > 1 else None,
                    }
                    for name in selected[0]["class_names"]
                },
            }
    return {"metric_scope": metric_key, "rows": rows}


def _summary_path(signature: str, stage: str) -> Path:
    return ROOT / "artifacts_v2" / "studies" / "followup" / f"retention_{stage}_{signature}_summary.json"


def _require_formal_prerequisites(base_signature: str, followup_signature: str) -> None:
    paths = [
        ROOT / "artifacts_v2" / "studies" / "benchmark" / f"formal_pu_{base_signature}_summary.json",
        _summary_path(followup_signature, "development"),
    ]
    for path in paths:
        if not path.is_file() or not json.loads(path.read_text(encoding="utf-8")).get("complete"):
            raise RuntimeError(f"formal retention curve requires completed prerequisite: {path}")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--protocol", type=Path, default=DEFAULT_FOLLOWUP)
    cli.add_argument("--stage", choices=("development", "formal"), required=True)
    cli.add_argument("--config", type=Path, default=ROOT / "configs" / "default_config.yaml")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--confirm-followup-signature", default="")
    cli.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "physical artifacts_v2 root for pending cells; completed identical cells are "
            "still reused from the repository artifacts_v2 root"
        ),
    )
    args = cli.parse_args()
    protocol, signature, base = load_followup(args.protocol)
    design = curve_design(protocol, args.stage)
    if args.stage == "formal":
        if args.confirm_followup_signature != signature:
            raise SystemExit(f"formal curve blocked; pass --confirm-followup-signature {signature}")
        _require_formal_prerequisites(protocol["base_protocol_signature"], signature)
    cells = [(seed, method, ratio) for seed in design["seeds"] for method in design["methods"] for ratio in design["ratios"]]
    output = _summary_path(signature, args.stage)
    summary = {
        "schema_version": 1, "complete": False, "status": "initializing",
        "stage": args.stage, "protocol_signature": signature,
        "protocol_path": str(args.protocol.resolve()), "protocol_sha256": file_sha256(args.protocol),
        "design": design, "independent_test_opened": False, "runs": {},
    }
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("protocol_signature") != signature or previous.get("stage") != args.stage:
            raise RuntimeError("incompatible retention summary")
        summary = previous
    primary_artifact_root = Path(args.artifact_root or (ROOT / "artifacts_v2")).resolve()
    fallback_artifact_root = (ROOT / "artifacts_v2").resolve()
    storage = summary.setdefault("storage", {})
    storage["pending_cell_write_root"] = str(primary_artifact_root)
    storage["reuse_roots"] = list(dict.fromkeys(
        [str(primary_artifact_root), str(fallback_artifact_root)]
    ))
    storage["location_affects_scientific_signature"] = False
    completed = {
        (int(seed), method, float(ratio))
        for seed, methods in summary.get("runs", {}).items()
        for method, ratios in methods.items()
        for ratio in ratios
    }
    pending = [cell for cell in cells if (cell[0], cell[1], round(cell[2], 2)) not in {(s, m, round(r, 2)) for s, m, r in completed}]
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "followup_signature": signature, "total_cells": len(cells), "completed_cells": len(completed), "pending_cells": [{"seed": s, "method": m, "ratio": r} for s, m, r in pending], "storage": storage, "independent_test_will_open": args.stage == "formal"}, indent=2))
        return
    cfg = load_config(args.config)
    for index, (seed, method, ratio) in enumerate(cells, 1):
        ratio_key = f"{ratio:.2f}"
        if ratio_key in summary.get("runs", {}).get(str(seed), {}).get(method, {}):
            print(f"[{args.stage} curve {index}/{len(cells)}] seed={seed}, {method}, r={ratio_key}: complete; skipping")
            continue
        print(f"\n[{args.stage} curve {index}/{len(cells)}] seed={seed}, method={method}, ratio={ratio_key}")
        condition_design = {**design, "retention_ratio": ratio}
        condition_args = _condition_args(base, condition_design, method, seed)
        condition_args.ratio = ratio
        condition_args.artifact_root = primary_artifact_root
        condition_args.reuse_artifact_roots = (fallback_artifact_root,)
        started = time.perf_counter()
        raw = run_condition(condition_args, cfg)
        run = _validate_and_enrich(raw, base, condition_design, method, seed, time.perf_counter() - started)
        baseline = summary.get("runs", {}).get(str(seed), {}).get(method, {})
        if baseline and any(value["split_fingerprint"] != run["split_fingerprint"] for value in baseline.values()):
            raise RuntimeError(f"split mismatch across ratios for {method}/seed{seed}")
        summary["runs"].setdefault(str(seed), {}).setdefault(method, {})[ratio_key] = run
        summary["aggregate"] = aggregate_curve(summary)
        summary["status"] = "running"
        summary["independent_test_opened"] = args.stage == "formal"
        export_json(output, summary)
    completed_count = sum(len(ratios) for methods in summary["runs"].values() for ratios in methods.values())
    summary["aggregate"] = aggregate_curve(summary)
    summary["complete"] = completed_count == len(cells)
    summary["status"] = "complete" if summary["complete"] else "incomplete"
    summary["paper_claim_authorized"] = bool(args.stage == "formal" and summary["complete"])
    export_json(output, summary)
    print(json.dumps({"status": summary["status"], "summary": str(output), "followup_signature": signature}, indent=2))


if __name__ == "__main__":
    main()
