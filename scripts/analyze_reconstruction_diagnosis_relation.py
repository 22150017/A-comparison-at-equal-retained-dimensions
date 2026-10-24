






from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_tools import export_json
from src.utils.reproducibility import file_sha256


def _rows(payload: dict, quality_sidecars: dict | None = None) -> list[dict]:
    if not payload.get("complete") or payload.get("stage") != "formal":
        raise ValueError("complete formal follow-up retention summary is required")
    if not payload.get("paper_claim_authorized"):
        raise ValueError("retention summary is not paper-authorized")
    rows = []
    for seed, methods in payload.get("runs", {}).items():
        for ratio, run in methods.get("ilsomp", {}).items():
            manifest_path = Path(run["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("complete") or file_sha256(manifest_path) != run["manifest_sha256"]:
                raise ValueError("Incomplete or changed reconstruction manifest")
            test_records = [record for record in manifest.get("records", []) if record.get("split") == "test"]
            values = [
                float(record["snr_db"])
                for record in test_records if record.get("snr_db") is not None
            ]
            if len(values) != len(test_records) and quality_sidecars is not None:
                sidecar = quality_sidecars.get("summaries", {}).get(f"seed{seed}_r{ratio.replace('.', '')}")
                if sidecar:
                    records_path = Path(sidecar["records_jsonl"])
                    if not quality_sidecars.get("complete") or not sidecar.get("complete"):
                        raise ValueError("Incomplete reconstruction sidecar")
                    if sidecar["manifest_sha256"] != file_sha256(manifest_path) or sidecar["records_sha256"] != file_sha256(records_path):
                        raise ValueError("Reconstruction sidecar provenance mismatch")
                    records = _load_quality_records(records_path)
                    lookup = {record["sample_id"]: record for record in records}
                    expected_ids = [record["sample_id"] for record in test_records]
                    if len(lookup) != len(records) or len(set(expected_ids)) != len(expected_ids) or set(lookup) != set(expected_ids):
                        raise ValueError("Reconstruction sidecar sample IDs mismatch")
                    values = [float(lookup[sample_id]["snr_db"]) for sample_id in expected_ids]
            if not test_records or len(values) != len(test_records) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"incomplete test reconstruction SNR values for seed={seed}, ratio={ratio}: "
                    f"{len(values)}/{len(test_records)}"
                )
            rows.append({
                "seed": int(seed), "retention_ratio": float(ratio),
                "reconstruction_snr_db_mean": float(np.mean(values)),
                "reconstruction_snr_db_std": float(np.std(values, ddof=1)),
                "diagnostic_macro_f1": float(run["paper_metrics"]["macro_f1"]),
                "sample_count": len(values), "manifest_sha256": file_sha256(manifest_path),
            })
    expected = len(payload["design"]["seeds"]) * len(payload["design"]["ratios"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} paired points, found {len(rows)}")
    return sorted(rows, key=lambda row: (row["retention_ratio"], row["seed"]))


def _load_quality_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(summary_path: Path, output_dir: Path, quality_sidecar_path: Path | None = None) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    sidecars = json.loads(quality_sidecar_path.read_text(encoding="utf-8")) if quality_sidecar_path else None
    rows = _rows(payload, sidecars)
    snr = np.asarray([row["reconstruction_snr_db_mean"] for row in rows])
    f1 = np.asarray([row["diagnostic_macro_f1"] for row in rows])
    
    
    
    descriptive_rho = (
        float(spearmanr(snr, f1).statistic)
        if np.ptp(snr) > 0 and np.ptp(f1) > 0
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ilsomp_reconstruction_snr_vs_macro_f1.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    scatter = ax.scatter(
        snr, f1,
        c=[row["retention_ratio"] for row in rows],
        cmap="viridis", edgecolor="black", linewidth=0.5,
    )
    ax.set_xlabel("Mean reconstruction SNR (dB)")
    ax.set_ylabel("Independent-test Macro-F1")
    ax.grid(alpha=0.20)
    fig.colorbar(scatter, ax=ax, label="Retention ratio")
    ax.text(
        0.03, 0.97,
        "Descriptive run-level co-trend\n4 fixed ratios x 3 seeds",
        transform=ax.transAxes, va="top",
    )
    fig.tight_layout()
    files = [csv_path]
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"ilsomp_reconstruction_snr_vs_macro_f1.{suffix}"
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight"); files.append(path)
    plt.close(fig)
    report = {
        "complete": True, "source_summary": str(summary_path.resolve()),
        "source_sha256": file_sha256(summary_path), "paired_point_count": len(rows),
        "descriptive_spearman_rho": descriptive_rho,
        "inferential_pvalue_evaluated": False,
        "quality_metric": "reconstruction_snr_db",
        "claim_boundary": (
            "Descriptive co-trend only. The 12 run-level points are repeated seed results at four fixed retention "
            "ratios, not independent experimental units; retention ratio is a shared driver/confound. No inferential "
            "p-value or causal claim is authorized. PSNR is not claimed because normalized vibration signals have no "
            "fixed physical peak range."
        ),
        "files": [str(path) for path in files],
    }
    report_path = output_dir / "ilsomp_reconstruction_snr_vs_macro_f1_report.json"
    export_json(report_path, report); report["files"].append(str(report_path))
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--summary", type=Path, required=True)
    cli.add_argument("--quality-sidecars", type=Path)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "artifacts_v2" / "reports" / "followup")
    args = cli.parse_args(); print(json.dumps(build(args.summary, args.output_dir, args.quality_sidecars), indent=2))


if __name__ == "__main__":
    main()
