
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_reconstruction_diagnosis_relation import build as build_relation
from src.utils.io_tools import export_json
from src.utils.reproducibility import file_sha256


def _validate_curve(payload: dict) -> list[dict]:
    if not payload.get("complete") or payload.get("stage") != "formal" or not payload.get("paper_claim_authorized"):
        raise ValueError("complete paper-authorized formal retention curve is required")
    design = payload["design"]
    rows = []
    for method in design["methods"]:
        for ratio in design["ratios"]:
            key = f"{method}@{ratio:.2f}"
            item = payload.get("aggregate", {}).get("rows", {}).get(key)
            if not item or item.get("n") != 3 or item["macro_f1"].get("std") is None:
                raise ValueError(f"formal curve cell lacks n=3 statistics: {key}")
            row = {
                "method": method, "retention_ratio": ratio, "n": item["n"],
                "macro_f1_mean": item["macro_f1"]["mean"], "macro_f1_std": item["macro_f1"]["std"],
                "accuracy_mean": item["accuracy"]["mean"], "accuracy_std": item["accuracy"]["std"],
            }
            for class_name, stats in item["per_class_recall"].items():
                row[f"{class_name}_recall_mean"] = stats["mean"]
                row[f"{class_name}_recall_std"] = stats["std"]
            rows.append(row)
    return rows


def build(curve_path: Path, efficiency_path: Path, quality_sidecar_path: Path, output_dir: Path) -> dict:
    curve = json.loads(curve_path.read_text(encoding="utf-8")); rows = _validate_curve(curve)
    efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
    if not efficiency.get("complete") or efficiency.get("independent_test_opened") is not False:
        raise ValueError("complete training-only efficiency benchmark is required")
    output_dir.mkdir(parents=True, exist_ok=True); figure_dir = output_dir / "figures"; figure_dir.mkdir(exist_ok=True)
    curve_csv = output_dir / "pu_ilsomp_pca_retention_curve.csv"
    with curve_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    efficiency_rows = []
    for method, item in efficiency["results"].items():
        efficiency_rows.append({
            "method": method, "measurement_count": item["measurement_count"],
            "transform_latency_ms_mean": item["transform_latency_ms"]["mean"],
            "transform_latency_ms_std": item["transform_latency_ms"]["std"],
            "transform_plus_stft_ms_mean": item["transform_plus_stft_latency_ms"]["mean"],
            "transform_plus_stft_ms_std": item["transform_plus_stft_latency_ms"]["std"],
            "pca_offline_fit_excluded": item["offline_fit_excluded"],
        })
    efficiency_csv = output_dir / "pu_method_efficiency.csv"
    with efficiency_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(efficiency_rows[0])); writer.writeheader(); writer.writerows(efficiency_rows)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method, color in (("ilsomp", "#0072B2"), ("pca", "#D55E00")):
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["retention_ratio"])
        ax.errorbar([row["retention_ratio"] for row in selected], [row["macro_f1_mean"] for row in selected], yerr=[row["macro_f1_std"] for row in selected], marker="o", capsize=3, label=method.upper(), color=color)
    ax.set_xlabel("Measurement retention ratio"); ax.set_ylabel("Independent-test Macro-F1"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25); ax.legend(frameon=False); fig.tight_layout()
    figures = []
    for suffix in ("png", "pdf", "svg"):
        path = figure_dir / f"pu_ilsomp_pca_retention_curve.{suffix}"; fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight"); figures.append(path)
    plt.close(fig)
    relation = build_relation(curve_path, output_dir / "reconstruction_relation", quality_sidecar_path)
    report = {
        "complete": True,
        "sources": {"curve": str(curve_path.resolve()), "efficiency": str(efficiency_path.resolve()), "quality_sidecars": str(quality_sidecar_path.resolve())},
        "source_sha256": {"curve": file_sha256(curve_path), "efficiency": file_sha256(efficiency_path), "quality_sidecars": file_sha256(quality_sidecar_path)},
        "claim_boundary": "Curve claims require paired n=3 formal cells. Efficiency is device-specific. Reconstruction-SNR association is non-causal and is not PSNR.",
        "files": [str(curve_csv), str(efficiency_csv), *map(str, figures), *relation["files"]],
    }
    report_path = output_dir / "followup_evidence_report.json"; export_json(report_path, report); report["files"].append(str(report_path))
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--curve-summary", type=Path, required=True)
    cli.add_argument("--efficiency-summary", type=Path, required=True)
    cli.add_argument("--quality-sidecars", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "artifacts_v2" / "reports" / "followup")
    args = cli.parse_args(); print(json.dumps(build(args.curve_summary, args.efficiency_summary, args.quality_sidecars, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
