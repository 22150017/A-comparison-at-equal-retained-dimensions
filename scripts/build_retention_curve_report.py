
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pubfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_tools import export_json
from src.utils.reproducibility import file_sha256


CLASS_NAMES = ("ball", "inner_race", "normal", "outer_race")
EXPECTED_RATIOS = ("0.50", "0.70", "0.80", "0.90")


def _discover(path: Path | None) -> Path:
    if path is not None:
        return path
    candidates = sorted((ROOT / "artifacts_v2" / "studies" / "pu").glob("retention_curve_3seed_*_summary.json"))
    complete = [item for item in candidates if json.loads(item.read_text(encoding="utf-8")).get("complete")]
    if len(complete) != 1:
        raise ValueError("expected exactly one complete retention curve; pass --summary explicitly")
    return complete[0]


def _validate(payload: dict) -> None:
    aggregate = payload.get("aggregate", {})
    if not payload.get("complete") or payload.get("status") != "complete":
        raise ValueError("retention curve is incomplete; paper figures are blocked")
    if tuple(aggregate.get("complete_ratios", ())) != EXPECTED_RATIOS:
        raise ValueError("retention curve lacks a complete formal ratio")
    for ratio in EXPECTED_RATIOS:
        row = aggregate.get("ratios", {}).get(ratio, {})
        if row.get("n") != 3:
            raise ValueError(f"retention ratio {ratio} does not have n=3")
    if aggregate.get("no_cs", {}).get("n") != 3:
        raise ValueError("no-CS reference does not have n=3")


def _rows(payload: dict) -> list[dict]:
    aggregate = payload["aggregate"]
    rows = []
    for ratio in EXPECTED_RATIOS:
        entry = aggregate["ratios"][ratio]
        row = {
            "retention_ratio": float(ratio), "n": entry["n"],
            "macro_f1_mean": entry["macro_f1"]["mean"], "macro_f1_std": entry["macro_f1"]["std"],
            "accuracy_mean": entry["accuracy"]["mean"], "accuracy_std": entry["accuracy"]["std"],
            "macro_f1_delta_vs_no_cs_mean": entry["macro_f1_delta_vs_no_cs"]["mean"],
            "macro_f1_delta_vs_no_cs_std": entry["macro_f1_delta_vs_no_cs"]["std"],
        }
        for name in CLASS_NAMES:
            row[f"{name}_recall_mean"] = entry["per_class_recall"][name]["mean"]
            row[f"{name}_recall_std"] = entry["per_class_recall"][name]["std"]
        rows.append(row)
    return rows


def _write_markdown(path: Path, rows: list[dict], baseline: dict) -> None:
    lines = [
        "# Three-seed ILSOMP retention boundary",
        "",
        "All values are independent-test mean ± sample standard deviation across seeds 42, 100, and 200 under the locked PU/STFT/source-file protocol.",
        "",
        f"No-CS reference: Macro-F1 {baseline['macro_f1']['mean']:.4f} ± {baseline['macro_f1']['std']:.4f}; Accuracy {baseline['accuracy']['mean']:.4f} ± {baseline['accuracy']['std']:.4f} (n=3).",
        "",
        "| Retention | Macro-F1 | Accuracy | Δ Macro-F1 vs no-CS | Ball recall | Inner recall | Normal recall | Outer recall |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['retention_ratio']:.2f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
            f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"{row['macro_f1_delta_vs_no_cs_mean']:.4f} ± {row['macro_f1_delta_vs_no_cs_std']:.4f} | "
            f"{row['ball_recall_mean']:.4f} ± {row['ball_recall_std']:.4f} | "
            f"{row['inner_race_recall_mean']:.4f} ± {row['inner_race_recall_std']:.4f} | "
            f"{row['normal_recall_mean']:.4f} ± {row['normal_recall_std']:.4f} | "
            f"{row['outer_race_recall_mean']:.4f} ± {row['outer_race_recall_std']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_figures(output_dir: Path, rows: list[dict], baseline: dict) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = np.asarray([row["retention_ratio"] for row in rows])
    means = np.asarray([row["macro_f1_mean"] for row in rows])
    stds = np.asarray([row["macro_f1_std"] for row in rows])
    fig = pubfig.line(
        means, x=ratios, series_names=["ILSOMP"], x_label="Measurement retention ratio",
        y_label="Independent-test Macro-F1", marker="o", title=None, theme=pubfig.get_theme("nature"), width=640,
    )
    axis = fig.axes[0]
    axis.errorbar(ratios, means, yerr=stds, fmt="none", capsize=3, color="#1f77b4", linewidth=1.0, label="±1 SD (n=3)")
    baseline_mean, baseline_std = baseline["macro_f1"]["mean"], baseline["macro_f1"]["std"]
    axis.axhline(baseline_mean, color="#333333", linestyle="--", linewidth=1.1, label="No-CS mean")
    axis.fill_between([ratios.min() - 0.02, ratios.max() + 0.02], baseline_mean - baseline_std, baseline_mean + baseline_std, color="#333333", alpha=0.10, linewidth=0)
    axis.set_xlim(ratios.min() - 0.03, ratios.max() + 0.03)
    axis.set_ylim(max(0.0, min(means - stds) - 0.08), min(1.01, baseline_mean + baseline_std + 0.02))
    axis.legend(frameon=False, loc="lower right")
    paths = pubfig.batch_export(
        fig,
        output_dir / "pu_ilsomp_retention_macro_f1",
        formats=("pdf", "svg", "png"),
        spec="nature",
        width="single",
    )
    plt.close(fig)
    return list(paths)


def build_report(summary_path: Path, output_dir: Path) -> dict:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate(payload)
    rows = _rows(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pu_ilsomp_retention_curve_3seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    markdown_path = output_dir / "pu_ilsomp_retention_curve_3seed.md"
    _write_markdown(markdown_path, rows, payload["aggregate"]["no_cs"])
    figure_paths = _export_figures(output_dir / "figures", rows, payload["aggregate"]["no_cs"])
    report = {
        "complete": True,
        "source_summary": str(summary_path.resolve()), "source_summary_sha256": file_sha256(summary_path),
        "claim_boundary": "Publication assets are authorized only for this complete n=3 retention characterization; no learned-sensing or cross-bearing screen result is included.",
        "files": [str(csv_path), str(markdown_path), *map(str, figure_paths)],
    }
    export_json(output_dir / "pu_ilsomp_retention_curve_3seed_report.json", report)
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--summary", type=Path)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "artifacts_v2" / "reports")
    args = cli.parse_args()
    report = build_report(_discover(args.summary), args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
