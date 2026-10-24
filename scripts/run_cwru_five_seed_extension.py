





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

from scripts.run_clean_study import load_config, load_samples, run_condition
from scripts.run_leakage_aware_benchmark import (
    _audit_resumed_runs,
    _condition_args,
    _format_duration,
    _load_protocol,
    _preflight_grouped_dataset,
    _stage_design,
    _validate_and_enrich,
    aggregate,
)
from src.utils.io_tools import export_json
from src.utils.reproducibility import canonical_hash, file_sha256

DEFAULT_EXTENSION = ROOT / "configs" / "cwru_five_seed_extension_v1.json"


def load_extension(path: Path) -> tuple[dict, dict, str, Path, Path]:
    extension = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "study_name", "base_protocol_path", "base_protocol_signature",
        "base_stage", "original_seeds", "extension_seeds", "methods",
        "retention_ratio", "split_unit", "reason",
    }
    missing = required - set(extension)
    if missing:
        raise ValueError(f"CWRU extension protocol lacks fields: {sorted(missing)}")
    if extension["base_stage"] != "external-cwru" or extension["split_unit"] != "source-file":
        raise ValueError("CWRU extension must preserve the external source-file protocol")
    if set(extension["original_seeds"]) & set(extension["extension_seeds"]):
        raise ValueError("original and extension seeds overlap")
    if extension["extension_seeds"] != [300, 400]:
        raise ValueError("v1 extension seeds are preregistered as 300 and 400")
    base_protocol_path = ROOT / str(extension["base_protocol_path"])
    base_protocol, base_signature = _load_protocol(base_protocol_path)
    if base_signature != extension["base_protocol_signature"]:
        raise RuntimeError("base protocol signature changed")
    if list(base_protocol["formal_seeds"]) != list(extension["original_seeds"]):
        raise RuntimeError("extension original seeds do not match the locked benchmark")
    if list(base_protocol["methods"]) != list(extension["methods"]):
        raise RuntimeError("extension methods do not match the locked benchmark")
    if float(base_protocol["retention_ratio"]) != float(extension["retention_ratio"]):
        raise RuntimeError("extension retention ratio does not match the locked benchmark")
    base_summary = (
        ROOT / "artifacts_v2" / "studies" / "benchmark"
        / f"external_cwru_{base_signature}_summary.json"
    )
    if not base_summary.is_file():
        raise FileNotFoundError(f"missing locked CWRU summary: {base_summary}")
    base = json.loads(base_summary.read_text(encoding="utf-8"))
    if (
        not base.get("complete")
        or not base.get("paper_claim_authorized")
        or base.get("stage") != "external-cwru"
    ):
        raise RuntimeError("locked three-seed CWRU benchmark is incomplete")
    design = _stage_design(base_protocol, "external-cwru")
    _audit_resumed_runs(base, base_protocol, design)
    signature = canonical_hash({
        "extension_protocol": extension,
        "extension_protocol_sha256": file_sha256(path),
        "base_protocol_sha256": file_sha256(base_protocol_path),
        "base_summary_sha256": file_sha256(base_summary),
    }, length=24)
    return extension, base_protocol, signature, base_summary, base_protocol_path


def _output_path(signature: str) -> Path:
    return (
        ROOT / "artifacts_v2" / "studies" / "benchmark"
        / f"external_cwru_five_seed_{signature}_summary.json"
    )


def _extension_design(base_protocol: dict, extension: dict) -> dict:
    design = _stage_design(base_protocol, "external-cwru")
    design["seeds"] = [int(seed) for seed in extension["extension_seeds"]]
    return design


def _merged_design(base_protocol: dict, extension: dict) -> dict:
    design = _stage_design(base_protocol, "external-cwru")
    design["seeds"] = [
        *[int(seed) for seed in extension["original_seeds"]],
        *[int(seed) for seed in extension["extension_seeds"]],
    ]
    return design


def _merged_runs(base: dict, extension_runs: dict) -> dict:
    runs = {seed: dict(methods) for seed, methods in base["runs"].items()}
    for seed, methods in extension_runs.items():
        if seed in runs:
            raise RuntimeError(f"extension would overwrite original seed {seed}")
        runs[seed] = dict(methods)
    return runs


def _preflight(base_protocol: dict, extension: dict, config_path: Path | None = None) -> dict:
    cfg = load_config(config_path or ROOT / "configs" / "default_config.yaml")
    samples = load_samples(cfg, "cwru")
    audit = _preflight_grouped_dataset(
        samples,
        float(cfg["training"]["train_ratio"]),
        float(cfg["training"]["val_ratio"]),
        [int(seed) for seed in extension["extension_seeds"]],
    )
    return {
        "status": "ready",
        "scope": "metadata-only CWRU seed-extension preflight",
        "dataset": "cwru",
        "methods": list(base_protocol["methods"]),
        "extension_seeds": list(extension["extension_seeds"]),
        **audit,
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--extension-protocol", type=Path, default=DEFAULT_EXTENSION)
    cli.add_argument("--config", type=Path, default=ROOT / "configs" / "default_config.yaml")
    cli.add_argument("--show-protocol", action="store_true")
    cli.add_argument("--preflight", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--confirm-extension-signature", default="")
    cli.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "physical artifacts_v2 root for newly prepared cells; existing matching "
            "cells may be reused from the repository artifacts_v2 root"
        ),
    )
    args = cli.parse_args()
    extension, base_protocol, signature, base_summary_path, base_protocol_path = load_extension(
        args.extension_protocol
    )
    if args.show_protocol:
        print(json.dumps({
            "extension_signature": signature,
            "extension_protocol": extension,
            "base_protocol": str(base_protocol_path),
            "base_summary": str(base_summary_path),
            "base_summary_sha256": file_sha256(base_summary_path),
        }, indent=2, ensure_ascii=False))
        return
    if args.preflight:
        print(json.dumps(_preflight(base_protocol, extension, args.config), indent=2, ensure_ascii=False))
        return
    if args.confirm_extension_signature != signature and not args.dry_run:
        raise SystemExit(
            "formal extension blocked: inspect --show-protocol, then pass "
            f"--confirm-extension-signature {signature}"
        )

    base = json.loads(base_summary_path.read_text(encoding="utf-8"))
    extension_design = _extension_design(base_protocol, extension)
    merged_design = _merged_design(base_protocol, extension)
    output = _output_path(signature)
    stage_signature = canonical_hash({
        "extension_signature": signature,
        "extension_design": extension_design,
        "base_summary_sha256": file_sha256(base_summary_path),
    }, length=24)
    summary = {
        "schema_version": 1,
        "complete": False,
        "status": "initializing",
        "stage": "external-cwru-five-seed-extension",
        "stage_signature": stage_signature,
        "extension_protocol_path": str(args.extension_protocol.resolve()),
        "extension_protocol_sha256": file_sha256(args.extension_protocol),
        "extension_signature": signature,
        "base_protocol_signature": extension["base_protocol_signature"],
        "base_summary": str(base_summary_path.resolve()),
        "base_summary_sha256": file_sha256(base_summary_path),
        "stage_design": merged_design,
        "extension_design": extension_design,
        "independent_test_opened": False,
        "extension_runs": {},
        "runs": dict(base["runs"]),
        "paper_claim_authorized": False,
    }
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("stage_signature") != stage_signature:
            raise RuntimeError(f"refusing incompatible extension summary: {output}")
        if previous.get("base_summary_sha256") != file_sha256(base_summary_path):
            raise RuntimeError("locked base CWRU summary changed after extension started")
        summary = previous
        _audit_resumed_runs(
            {"runs": summary.get("extension_runs", {})},
            base_protocol,
            extension_design,
        )
        print(f"resuming CWRU extension: {output}")

    cells = [
        (int(seed), method)
        for seed in extension["extension_seeds"]
        for method in extension["methods"]
    ]
    completed = {
        (int(seed), method)
        for seed, methods in summary.get("extension_runs", {}).items()
        for method in methods
    }
    pending = [cell for cell in cells if cell not in completed]
    if args.dry_run:
        print(json.dumps({
            "stage": summary["stage"],
            "extension_signature": signature,
            "base_three_seed_complete": True,
            "original_cells_preserved": 15,
            "extension_total_cells": len(cells),
            "extension_completed_cells": len(completed),
            "pending_cells": [
                {"seed": seed, "method": method} for seed, method in pending
            ],
            "final_seed_count": 5,
            "independent_test_will_open": True,
        }, indent=2, ensure_ascii=False))
        return

    cfg = load_config(args.config)
    primary_artifact_root = Path(args.artifact_root or (ROOT / "artifacts_v2")).resolve()
    fallback_artifact_root = (ROOT / "artifacts_v2").resolve()
    summary.setdefault("storage", {})
    summary["storage"].update({
        "pending_cell_write_root": str(primary_artifact_root),
        "reuse_roots": list(dict.fromkeys(
            [str(primary_artifact_root), str(fallback_artifact_root)]
        )),
        "location_affects_scientific_signature": False,
    })
    observed = [
        float(run.get("runner_elapsed_seconds", 0.0))
        for methods in summary.get("extension_runs", {}).values()
        for run in methods.values()
        if float(run.get("runner_elapsed_seconds", 0.0)) >= 60.0
    ]
    for index, (seed, method) in enumerate(cells, 1):
        if (seed, method) in completed:
            print(f"[CWRU-5] seed={seed}, method={method}: complete; skipping")
            continue
        mean_seconds = float(np.mean(observed)) if observed else None
        remaining_after = len(cells) - index
        print(
            f"\n[CWRU-5 {index}/{len(cells)}] seed={seed}, method={method}; "
            f"ETA after current={_format_duration(mean_seconds * remaining_after if mean_seconds else None)}"
        )
        started = time.perf_counter()
        condition_args = _condition_args(base_protocol, extension_design, method, seed)
        condition_args.artifact_root = primary_artifact_root
        condition_args.reuse_artifact_roots = (fallback_artifact_root,)
        raw = run_condition(condition_args, cfg)
        elapsed = time.perf_counter() - started
        run = _validate_and_enrich(raw, base_protocol, extension_design, method, seed, elapsed)
        baseline = summary["extension_runs"].get(str(seed), {}).get("no_cs")
        if baseline is not None and run["split_fingerprint"] != baseline["split_fingerprint"]:
            raise RuntimeError(f"split mismatch for CWRU extension seed={seed}: no_cs vs {method}")
        summary["extension_runs"].setdefault(str(seed), {})[method] = run
        summary["runs"] = _merged_runs(base, summary["extension_runs"])
        summary["aggregate"] = aggregate({"stage_design": merged_design, "runs": summary["runs"]})
        summary["status"] = "running"
        summary["independent_test_opened"] = True
        export_json(output, summary)
        if elapsed >= 60.0:
            observed.append(elapsed)

    completed = {
        (int(seed), method)
        for seed, methods in summary.get("extension_runs", {}).items()
        for method in methods
    }
    summary["runs"] = _merged_runs(base, summary["extension_runs"])
    summary["aggregate"] = aggregate({"stage_design": merged_design, "runs": summary["runs"]})
    summary["complete"] = completed == set(cells)
    summary["status"] = "complete" if summary["complete"] else "incomplete"
    summary["independent_test_opened"] = bool(completed)
    summary["paper_claim_authorized"] = summary["complete"]
    summary["claim_boundary"] = (
        "Five-seed CWRU statistics combine the immutable original seeds 42/100/200 with the "
        "prospectively declared variance-motivated extension seeds 300/400. The extension was "
        "defined after observing high three-seed variance and must be disclosed as such."
    )
    export_json(output, summary)
    print(json.dumps({
        "status": summary["status"],
        "summary": str(output),
        "extension_signature": signature,
        "aggregate": summary["aggregate"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
