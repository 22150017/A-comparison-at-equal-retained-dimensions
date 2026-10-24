
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_tools import export_json
from src.utils.reproducibility import file_sha256


METHOD_LABELS = {
    "no_cs": "No compression",
    "ilsomp": "ILSOMP",
    "downsample": "Downsampling",
    "pca": "PCA",
    "random_projection": "Random projection",
}


def _discover(stage: str, explicit: Path | None, required: bool) -> Path | None:
    if explicit is not None:
        return explicit
    if stage == "external-cwru":
        extensions = sorted((ROOT / "artifacts_v2" / "studies" / "benchmark").glob("external_cwru_five_seed_*_summary.json"))
        complete_extensions = [path for path in extensions if json.loads(path.read_text(encoding="utf-8")).get("complete")]
        if len(complete_extensions) == 1:
            return complete_extensions[0]
        if len(complete_extensions) > 1:
            raise ValueError("multiple complete five-seed CWRU summaries; pass the path explicitly")
    pattern = f"{stage.replace('-', '_')}_*_summary.json"
    candidates = sorted((ROOT / "artifacts_v2" / "studies" / "benchmark").glob(pattern))
    complete = [path for path in candidates if json.loads(path.read_text(encoding="utf-8")).get("complete")]
    if not complete and not required:
        return None
    if len(complete) != 1:
        raise ValueError(f"expected exactly one complete {stage} summary; pass its path explicitly")
    return complete[0]


def _load_complete(path: Path, stage: str, paper_authorized: bool) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_stages = {stage}
    if stage == "external-cwru":
        allowed_stages.add("external-cwru-five-seed-extension")
    if payload.get("stage") not in allowed_stages or not payload.get("complete"):
        raise ValueError(f"incomplete or wrong-stage summary: {path}")
    design = payload.get("stage_design", {})
    expected_n = len(design.get("seeds", []))
    required_n = 5 if payload["stage"] == "external-cwru-five-seed-extension" else 3
    if expected_n != required_n:
        raise ValueError(f"{stage} must have {required_n} formal seeds")
    for method_name in design.get("methods", []):
        item = payload.get("aggregate", {}).get("methods", {}).get(method_name, {})
        if item.get("n") != expected_n:
            raise ValueError(f"{stage}/{method_name} does not have n={expected_n}")
    if paper_authorized and payload.get("paper_claim_authorized") is not True:
        raise ValueError(f"paper claims are not authorized by {path}")
    if stage == "leakage-audit" and payload.get("paper_claim_authorized") is not False:
        raise ValueError("leakage control must never authorize generalization performance")
    return payload


def _rows(payload: dict, evidence_role: str) -> list[dict]:
    rows = []
    for method_name in payload["stage_design"]["methods"]:
        item = payload["aggregate"]["methods"][method_name]
        delta = item.get("paired_macro_f1_delta_vs_no_cs", {})
        row = {
            "stage": payload["stage"],
            "evidence_role": evidence_role,
            "dataset": payload["stage_design"]["dataset"],
            "split_unit": payload["stage_design"]["split_unit"],
            "method": method_name,
            "retention_ratio": (
                1.0
                if method_name == "no_cs"
                else float(payload["stage_design"].get("retention_ratio", 0.5))
            ),
            "n": item["n"],
            "macro_f1_mean": item["macro_f1"]["mean"],
            "macro_f1_std": item["macro_f1"]["std"],
            "accuracy_mean": item["accuracy"]["mean"],
            "accuracy_std": item["accuracy"]["std"],
            "paired_macro_f1_delta_vs_no_cs_mean": delta.get("mean"),
            "paired_macro_f1_delta_vs_no_cs_std": delta.get("std"),
            "training_epoch_seconds_mean": item["training_epoch_seconds"]["mean"],
            "generalization_performance_authorized": evidence_role != "deliberately_leaky_control",
        }
        for class_name, stats in item.get("per_class_recall", {}).items():
            row[f"recall_{class_name}_mean"] = stats["mean"]
            row[f"recall_{class_name}_std"] = stats["std"]
        rows.append(row)
    return rows


def _plot_equal_budget(rows: list[dict], output_dir: Path) -> list[Path]:
    primary = [row for row in rows if row["stage"] == "formal-pu"]
    labels = [METHOD_LABELS[row["method"]] for row in primary]
    means = np.asarray([row["macro_f1_mean"] for row in primary])
    stds = np.asarray([row["macro_f1_std"] for row in primary])
    colors = ["#4D4D4D", "#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    positions = np.arange(len(primary))
    ax.bar(positions, means, yerr=stds, capsize=3, color=colors[:len(primary)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(positions, labels, rotation=18, ha="right")
    ax.set_ylabel("Independent-test Macro-F1")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.text(0.99, 0.02, "mean ± SD, n=3", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    base = output_dir / "pu_equal_budget_macro_f1"
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_leakage(rows: list[dict], output_dir: Path) -> list[Path]:
    valid = {row["method"]: row for row in rows if row["stage"] == "formal-pu"}
    leaky = {row["method"]: row for row in rows if row["stage"] == "leakage-audit"}
    methods = [method for method in ("no_cs", "ilsomp") if method in valid and method in leaky]
    if len(methods) != 2:
        return []
    x = np.arange(len(methods)); width = 0.34
    valid_means = [valid[method]["macro_f1_mean"] for method in methods]
    valid_stds = [valid[method]["macro_f1_std"] for method in methods]
    leaky_means = [leaky[method]["macro_f1_mean"] for method in methods]
    leaky_stds = [leaky[method]["macro_f1_std"] for method in methods]
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    ax.bar(x - width / 2, valid_means, width, yerr=valid_stds, capsize=3, label="Source-file split", color="#0072B2")
    ax.bar(x + width / 2, leaky_means, width, yerr=leaky_stds, capsize=3, label="Window-level leaky control", color="#D55E00")
    ax.set_xticks(x, [METHOD_LABELS[method] for method in methods])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    base = output_dir / "pu_split_protocol_inflation"
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Leakage-aware matched-budget benchmark",
        "",
        "Primary results use source-file-disjoint independent tests. Window-level results are deliberately leaky controls and must never be described as generalization performance.",
        "",
        "| Dataset | Split/evidence | Method | Retention | Macro-F1 | Accuracy | Paired ΔF1 vs no-CS | n |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["paired_macro_f1_delta_vs_no_cs_mean"]
        delta_std = row["paired_macro_f1_delta_vs_no_cs_std"]
        delta_text = "—" if delta is None else f"{delta:.4f} ± {delta_std:.4f}"
        lines.append(
            f"| {row['dataset']} | {row['evidence_role']} | {METHOD_LABELS[row['method']]} | "
            f"{row['retention_ratio']:.2f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
            f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {delta_text} | {row['n']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    formal_path: Path,
    output_dir: Path,
    leakage_path: Path | None = None,
    external_path: Path | None = None,
) -> dict:
    formal = _load_complete(formal_path, "formal-pu", paper_authorized=True)
    rows = _rows(formal, "paper_valid_primary")
    sources = {"formal_pu": str(formal_path.resolve())}
    if leakage_path is not None:
        leakage = _load_complete(leakage_path, "leakage-audit", paper_authorized=False)
        if leakage["protocol_signature"] != formal["protocol_signature"]:
            raise ValueError("leakage/formal protocol signatures differ")
        rows.extend(_rows(leakage, "deliberately_leaky_control"))
        sources["leakage_audit"] = str(leakage_path.resolve())
    if external_path is not None:
        external = _load_complete(external_path, "external-cwru", paper_authorized=True)
        external_base_signature = external.get("protocol_signature", external.get("base_protocol_signature"))
        if external_base_signature != formal["protocol_signature"]:
            raise ValueError("external/formal protocol signatures differ")
        rows.extend(_rows(external, "paper_valid_external_confirmation"))
        sources["external_cwru"] = str(external_path.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "leakage_aware_benchmark.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    markdown_path = output_dir / "leakage_aware_benchmark.md"
    _write_markdown(markdown_path, rows)
    figures = _plot_equal_budget(rows, figure_dir) + _plot_leakage(rows, figure_dir)
    report = {
        "complete": True,
        "protocol_signature": formal["protocol_signature"],
        "sources": sources,
        "source_sha256": {name: file_sha256(Path(path)) for name, path in sources.items()},
        "claim_boundary": (
            "Only source-file formal/external rows authorize generalization-performance claims; "
            "window-control rows quantify leakage inflation only."
        ),
        "files": [str(csv_path), str(markdown_path), *map(str, figures)],
    }
    report_path = output_dir / "leakage_aware_benchmark_report.json"
    export_json(report_path, report)
    report["files"].append(str(report_path))
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--formal-pu", type=Path)
    cli.add_argument("--leakage-audit", type=Path)
    cli.add_argument("--external-cwru", type=Path)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "artifacts_v2" / "reports" / "benchmark")
    args = cli.parse_args()
    formal_path = _discover("formal-pu", args.formal_pu, required=True)
    leakage_path = _discover("leakage-audit", args.leakage_audit, required=False)
    external_path = _discover("external-cwru", args.external_cwru, required=False)
    report = build_report(formal_path, args.output_dir, leakage_path, external_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
