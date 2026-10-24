
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
from scripts.validate_study_artifacts import validate_manifest, validate_metrics
from src.utils.io_tools import export_json
from src.utils.reproducibility import canonical_hash


CLASS_NAMES = ("ball", "inner_race", "normal", "outer_race")


def _csv_seeds(text: str) -> list[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    if len(seeds) < 3:
        raise ValueError("the formal paired study requires at least three distinct seeds")
    return seeds


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m" if hours else f"{minutes:d}m {secs:02d}s"


def _split_fingerprint(manifest: dict) -> str:
    records = manifest.get("records", [])
    assignments = []
    seen: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        split = str(record.get("split", ""))
        if not sample_id or split not in {"train", "val", "test"}:
            raise ValueError("manifest contains an invalid sample split assignment")
        if sample_id in seen:
            raise ValueError(f"manifest contains duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        assignments.append((sample_id, split))
    if not assignments:
        raise ValueError("manifest has no split assignments")
    return canonical_hash(sorted(assignments), length=24)


def _validate_arm_manifest(manifest: dict, arm: str, seed: int, ratio: float) -> str:
    condition = manifest.get("condition", {})
    split = condition.get("split", {})
    if not manifest.get("complete"):
        raise ValueError(f"{arm} manifest is incomplete")
    if condition.get("seed") != seed:
        raise ValueError(f"{arm} manifest seed mismatch")
    if condition.get("representation") != "stft":
        raise ValueError(f"{arm} must use STFT")
    if split.get("mode") != "stratified_source_file_random" or split.get("unit") != "source_file":
        raise ValueError(f"{arm} must use the calibrated source-file split")
    if arm == "no_cs":
        if condition.get("method") is not None or condition.get("compression_ratio") is not None:
            raise ValueError("no-CS arm contains compression metadata")
    elif arm == "ilsomp50":
        if condition.get("method") != "ilsomp" or not np.isclose(
            float(condition.get("compression_ratio", -1.0)), ratio
        ):
            raise ValueError("ILSOMP arm method or retention ratio mismatch")
    else:
        raise ValueError(f"unknown paired arm: {arm}")
    return _split_fingerprint(manifest)


def _per_class_recalls(metrics: dict) -> dict[str, float]:
    matrix = np.asarray(metrics.get("confusion_matrix"), dtype=float)
    if matrix.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError(f"expected a {len(CLASS_NAMES)}x{len(CLASS_NAMES)} confusion matrix")
    denominators = matrix.sum(axis=1)
    if np.any(denominators <= 0):
        raise ValueError("confusion matrix contains an empty test class")
    values = np.diag(matrix) / denominators
    return {name: float(value) for name, value in zip(CLASS_NAMES, values)}


def _baseline_gate(runs: list[dict]) -> dict:
    macro_f1 = np.asarray([run["paper_metrics"]["macro_f1"] for run in runs], dtype=float)
    recalls = [_per_class_recalls(run["paper_metrics"]) for run in runs]
    std = float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else None
    per_run_pass = [
        float(run["paper_metrics"]["macro_f1"]) >= 0.80
        and min(recall.values()) >= 0.70
        for run, recall in zip(runs, recalls)
    ]
    passed = bool(
        len(runs) >= 3
        and all(per_run_pass)
        and float(macro_f1.mean()) >= 0.80
        and std is not None
        and std <= 0.05
    )
    return {
        "passed": passed,
        "n": len(runs),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": std,
        "per_run_pass": per_run_pass,
        "per_class_recalls": recalls,
        "thresholds": {
            "minimum_runs": 3,
            "macro_f1_per_run_min": 0.80,
            "class_recall_per_run_min": 0.70,
            "macro_f1_mean_min": 0.80,
            "macro_f1_std_max": 0.05,
        },
    }


def _aggregate_pairs(pairs: dict[str, dict]) -> dict:
    complete = [pair for pair in pairs.values() if "no_cs" in pair and "ilsomp50" in pair]
    if not complete:
        return {"n": 0}
    baseline_accuracy = np.asarray(
        [pair["no_cs"]["paper_metrics"]["accuracy"] for pair in complete], dtype=float
    )
    cs_accuracy = np.asarray(
        [pair["ilsomp50"]["paper_metrics"]["accuracy"] for pair in complete], dtype=float
    )
    baseline_f1 = np.asarray(
        [pair["no_cs"]["paper_metrics"]["macro_f1"] for pair in complete], dtype=float
    )
    cs_f1 = np.asarray(
        [pair["ilsomp50"]["paper_metrics"]["macro_f1"] for pair in complete], dtype=float
    )

    def stats(values: np.ndarray) -> dict:
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        }

    return {
        "n": len(complete),
        "no_cs_accuracy": stats(baseline_accuracy),
        "ilsomp50_accuracy": stats(cs_accuracy),
        "paired_accuracy_delta_cs_minus_no_cs": stats(cs_accuracy - baseline_accuracy),
        "no_cs_macro_f1": stats(baseline_f1),
        "ilsomp50_macro_f1": stats(cs_f1),
        "paired_macro_f1_delta_cs_minus_no_cs": stats(cs_f1 - baseline_f1),
    }


def _condition_args(args: argparse.Namespace, seed: int, arm: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=args.dataset,
        method="ilsomp",
        ratio=args.ratio,
        representation="stft",
        backbone=args.backbone,
        seed=seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        freeze_ratio=args.freeze_ratio,
        num_workers=args.num_workers,
        amp=args.amp,
        class_weighting="balanced",
        sampler="none",
        report_train_metrics=False,
        checkpoint_selection="best-val",
        fast_training=args.fast_training,
        prefetch_factor=args.prefetch_factor,
        eval_num_workers=args.eval_num_workers,
        prepare_workers=args.prepare_workers,
        prepare_chunksize=args.prepare_chunksize,
        no_cs=arm == "no_cs",
        sparsity_k=args.sparsity_k,
        gradient_lambda=args.gradient_lambda,
        sparse_basis=args.sparse_basis,
        wavelet=None,
        cwt_scales_start=1,
        cwt_scales_end=128,
        stft_nperseg=args.stft_nperseg,
        stft_noverlap=args.stft_noverlap,
        noise_snr_db=None,
        train_conditions=(),
        test_condition=None,
        compression_baseline=None,
        pca_fit_samples=4096,
        test_groups="",
        val_groups="",
        split_unit="source-file",
        prepare_only=False,
        max_samples_per_group=None,
    )


def _run_arm(args: argparse.Namespace, cfg: dict, seed: int, arm: str) -> dict:
    started = time.perf_counter()
    run = run_condition(_condition_args(args, seed, arm), cfg)
    run["elapsed_seconds"] = time.perf_counter() - started
    manifest_path = Path(run["dataset_root"]) / "dataset_manifest.json"
    metrics_path = Path(run["result_root"]) / f"{args.backbone}_metrics.json"
    errors = validate_manifest(manifest_path) + validate_metrics(metrics_path)
    if errors:
        raise RuntimeError("; ".join(errors))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run["split_fingerprint"] = _validate_arm_manifest(manifest, arm, seed, args.ratio)
    run["artifact_validation"] = "passed"
    run["per_class_recalls"] = _per_class_recalls(run["paper_metrics"])
    return run


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=ROOT / "configs" / "default_config.yaml")
    cli.add_argument("--dataset", choices=("pu", "cwru"), default="pu")
    cli.add_argument("--seeds", default="42,100,200")
    cli.add_argument("--ratio", type=float, default=0.5)
    cli.add_argument("--backbone", default="resnet34_ema")
    cli.add_argument("--epochs", type=int, default=30)
    cli.add_argument("--learning-rate", type=float, default=1e-3)
    cli.add_argument("--batch-size", type=int, default=32)
    cli.add_argument("--freeze-ratio", type=float, default=0.5)
    cli.add_argument("--num-workers", type=int, default=2)
    cli.add_argument("--prepare-workers", type=int, default=2)
    cli.add_argument("--prepare-chunksize", type=int, default=8)
    cli.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--fast-training", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--prefetch-factor", type=int, default=4)
    cli.add_argument("--eval-num-workers", type=int, default=0)
    cli.add_argument("--sparsity-k", type=int, default=64)
    cli.add_argument("--gradient-lambda", type=float, default=0.25)
    cli.add_argument("--sparse-basis", choices=("dct", "db4", "db8"), default="dct")
    cli.add_argument("--stft-nperseg", type=int, default=64)
    cli.add_argument("--stft-noverlap", type=int, default=32)
    return cli


def main() -> None:
    args = parser().parse_args()
    if not np.isclose(args.ratio, 0.5):
        raise SystemExit("this preregistered runner requires --ratio 0.5")
    seeds = _csv_seeds(args.seeds)
    cfg = load_config(args.config)
    protocol_config = {
        key: str(value.resolve()) if key == "config" else value
        for key, value in vars(args).items()
    }
    protocol_config["seeds"] = seeds
    protocol_config["representation"] = "stft"
    protocol_config["split_unit"] = "source-file"
    protocol_config["class_weighting"] = "balanced"
    protocol_config["checkpoint_selection"] = "best-val"
    protocol_signature = canonical_hash(protocol_config)
    summary_path = (
        ROOT / "artifacts_v2" / "studies" / args.dataset
        / f"paired_ilsomp50_{protocol_signature}_summary.json"
    )
    summary = {
        "protocol_signature": protocol_signature,
        "protocol_config": protocol_config,
        "complete": False,
        "status": "running_baselines",
        "pairs": {},
    }
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("protocol_signature") != protocol_signature:
            raise RuntimeError(f"protocol signature mismatch: {summary_path}")
        summary = previous
        print(f"resuming paired summary: {summary_path}")

    observed_seconds = [
        float(run.get("elapsed_seconds", 0.0))
        for pair in summary.get("pairs", {}).values()
        for arm, run in pair.items()
        if arm in {"no_cs", "ilsomp50"} and float(run.get("elapsed_seconds", 0.0)) > 1.0
    ]
    for seed in seeds:
        pair = summary["pairs"].setdefault(str(seed), {})
        if "no_cs" not in pair:
            print(f"\n[paired baseline] seed={seed}")
            pair["no_cs"] = _run_arm(args, cfg, seed, "no_cs")
            observed_seconds.append(pair["no_cs"]["elapsed_seconds"])
            export_json(summary_path, summary)
        else:
            print(f"[paired baseline] seed={seed} already recorded; skipping")

    baseline_runs = [summary["pairs"][str(seed)]["no_cs"] for seed in seeds]
    summary["baseline_gate"] = _baseline_gate(baseline_runs)
    summary["baseline_gate_used_for_execution"] = False

    summary["status"] = "running_ilsomp50"
    export_json(summary_path, summary)
    total_cs = len(seeds)
    for position, seed in enumerate(seeds, 1):
        pair = summary["pairs"][str(seed)]
        if "ilsomp50" in pair:
            print(f"[paired ILSOMP-50] seed={seed} already recorded; skipping")
            continue
        mean_seconds = float(np.mean(observed_seconds)) if observed_seconds else None
        remaining = total_cs - position
        print(
            f"\n[paired ILSOMP-50 {position}/{total_cs}] seed={seed}; "
            f"ETA after current={_format_duration(mean_seconds * remaining if mean_seconds else None)}"
        )
        pair["ilsomp50"] = _run_arm(args, cfg, seed, "ilsomp50")
        if pair["no_cs"]["split_fingerprint"] != pair["ilsomp50"]["split_fingerprint"]:
            del pair["ilsomp50"]
            export_json(summary_path, summary)
            raise RuntimeError(f"paired split mismatch for seed {seed}")
        pair["paired_delta"] = {
            "accuracy_cs_minus_no_cs": float(
                pair["ilsomp50"]["paper_metrics"]["accuracy"]
                - pair["no_cs"]["paper_metrics"]["accuracy"]
            ),
            "macro_f1_cs_minus_no_cs": float(
                pair["ilsomp50"]["paper_metrics"]["macro_f1"]
                - pair["no_cs"]["paper_metrics"]["macro_f1"]
            ),
            "per_class_recall_cs_minus_no_cs": {
                name: float(
                    pair["ilsomp50"]["per_class_recalls"][name]
                    - pair["no_cs"]["per_class_recalls"][name]
                )
                for name in CLASS_NAMES
            },
        }
        observed_seconds.append(pair["ilsomp50"]["elapsed_seconds"])
        summary["aggregate"] = _aggregate_pairs(summary["pairs"])
        export_json(summary_path, summary)

    summary["aggregate"] = _aggregate_pairs(summary["pairs"])
    summary["complete"] = summary["aggregate"].get("n") == len(seeds)
    summary["status"] = "complete" if summary["complete"] else "incomplete"
    export_json(summary_path, summary)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
