from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from rie_supplement_support import METHODS, artifact, read_json, save_json
from validate_nde_noise_robustness_extension import validate


CONDITIONS = ("original", "20dB", "10dB")
LABELS = {"no_cs": "Uncompressed", "ilsomp": "ILSOMP", "downsample": "Downsampling", "pca": "PCA", "random_projection": "Random projection"}


def descriptive(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"values": list(map(float, data)), "n": len(data), "mean": float(data.mean()),
            "sample_sd": float(data.std(ddof=1)), "median": float(np.median(data)),
            "minimum": float(data.min()), "maximum": float(data.max()),
            "positive_runs": int(np.count_nonzero(data > 0)), "negative_runs": int(np.count_nonzero(data < 0)),
            "zero_runs": int(np.count_nonzero(data == 0)), "nonnegative_runs": int(np.count_nonzero(data >= 0))}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_analysis(summary: dict, protocol: dict) -> dict:
    source = []
    for cell in summary["clean_cells"]:
        source.append(("original", cell))
    for cell in summary["cells"]:
        source.append((f"{cell['snr_db']:g}dB", cell))
    rows, recall_rows, file_rows = [], [], []
    files_by_run = defaultdict(set)
    for condition, cell in source:
        metric = cell["metrics"]
        row = {"dataset": cell["dataset"], "seed": cell["seed"], "method": cell["method"], "condition": condition,
               "window_macro_f1": metric["window"]["macro_f1"], "minimum_class_recall": metric["window"]["minimum_class_recall"],
               "source_macro_f1": metric["source"]["macro_f1"], "failed_source_count": metric["failed_source_count"],
               "test_windows": metric["window_count"], "test_source_files": metric["source_file_count"]}
        rows.append(row)
        for name, recall in metric["window"]["per_class_recall"].items():
            recall_rows.append({key: row[key] for key in ("dataset", "seed", "method", "condition")} |
                               {"stored_class_name": name, "paper_class_name": "compound" if row["dataset"] == "pu" and name == "ball" else name, "recall": recall})
        for item in metric["source_rows"]:
            paper_target = "compound" if row["dataset"] == "pu" and item["class_name"] == "ball" else item["class_name"]
            paper_prediction = "compound" if row["dataset"] == "pu" and item["predicted_class"] == "ball" else item["predicted_class"]
            file_rows.append({key: row[key] for key in ("dataset", "seed", "method", "condition")} | item |
                             {"paper_class_name": paper_target, "paper_predicted_class": paper_prediction})
            if condition == "original" and cell["method"] == "no_cs":
                files_by_run[(cell["dataset"], cell["seed"])].add(item["source_file"])
    order = {name: i for i, name in enumerate(CONDITIONS)}
    rows.sort(key=lambda r: (r["dataset"], r["seed"], METHODS.index(r["method"]), order[r["condition"]]))
    lookup = {(r["dataset"], r["seed"], r["method"], r["condition"]): r for r in rows}
    values = defaultdict(list)
    for row in rows:
        values[(row["dataset"], row["method"], row["condition"])].append(row)
    aggregates = []
    metrics = ("window_macro_f1", "minimum_class_recall", "source_macro_f1", "failed_source_count")
    for (dataset, method, condition), group in values.items():
        aggregates.append({"dataset": dataset, "method": method, "condition": condition, "seeds": [r["seed"] for r in group],
                           "metrics": {name: descriptive([r[name] for r in group]) for name in metrics}})
    contrasts, changes, level_contrasts = [], [], []
    for dataset, seeds in protocol["design"]["datasets_and_seeds"].items():
        for seed in seeds:
            for condition in CONDITIONS:
                ref = lookup[(dataset, seed, "no_cs", condition)]
                for method in METHODS[1:]:
                    candidate = lookup[(dataset, seed, method, condition)]
                    contrasts.append({"dataset": dataset, "seed": seed, "method": method, "condition": condition} |
                                     {name + "_delta": candidate[name] - ref[name] for name in metrics})
                    level_contrasts.append({"dataset": dataset, "seed": seed, "method": method, "condition": condition,
                                            "window_delta": candidate["window_macro_f1"] - ref["window_macro_f1"],
                                            "source_delta": candidate["source_macro_f1"] - ref["source_macro_f1"],
                                            "window_lower_source_higher": candidate["window_macro_f1"] < ref["window_macro_f1"] and candidate["source_macro_f1"] > ref["source_macro_f1"],
                                            "window_higher_source_lower": candidate["window_macro_f1"] > ref["window_macro_f1"] and candidate["source_macro_f1"] < ref["source_macro_f1"]})
            for method in METHODS[1:]:
                original = lookup[(dataset, seed, method, "original")]
                original_ref = lookup[(dataset, seed, "no_cs", "original")]
                for condition in CONDITIONS[1:]:
                    current = lookup[(dataset, seed, method, condition)]
                    current_ref = lookup[(dataset, seed, "no_cs", condition)]
                    change = {"dataset": dataset, "seed": seed, "method": method, "condition": condition}
                    for name in metrics:
                        clean_delta = original[name] - original_ref[name]
                        noisy_delta = current[name] - current_ref[name]
                        change[name + "_original_delta"] = clean_delta
                        change[name + "_noisy_delta"] = noisy_delta
                        change[name + "_contrast_change"] = noisy_delta - clean_delta
                        if name != "failed_source_count":
                            change[name + "_strict_negative_to_positive"] = clean_delta < 0 and noisy_delta > 0
                            change[name + "_strict_positive_to_negative"] = clean_delta > 0 and noisy_delta < 0
                    changes.append(change)
    comparison_summary = []
    for dataset, seeds in protocol["design"]["datasets_and_seeds"].items():
        for method in METHODS[1:]:
            for condition in CONDITIONS:
                group = [r for r in contrasts if (r["dataset"], r["method"], r["condition"]) == (dataset, method, condition)]
                result = {"dataset": dataset, "method": method, "condition": condition, "seeds": seeds,
                          "metrics": {name: descriptive([r[name + "_delta"] for r in group]) for name in metrics}}
                if condition != "original":
                    shift = [r for r in changes if (r["dataset"], r["method"], r["condition"]) == (dataset, method, condition)]
                    result["contrast_changes"] = {name: descriptive([r[name + "_contrast_change"] for r in shift]) for name in metrics}
                    result["strict_window_reversal_runs"] = sum(r["window_macro_f1_strict_negative_to_positive"] for r in shift)
                comparison_summary.append(result)
    overlap = {}
    for dataset in ("pu", "cwru"):
        groups = [files for (name, _), files in files_by_run.items() if name == dataset]
        union = set().union(*groups)
        overlap[dataset] = {"runs": len(groups), "source_files_per_run": [len(g) for g in groups],
                            "source_file_occurrences": sum(map(len, groups)), "unique_source_paths_across_runs": len(union),
                            "repeated_source_occurrences": sum(map(len, groups)) - len(union),
                            "physical_bearing_independent_replications_claimed": False}
    return {"rows": rows, "class_recalls": recall_rows, "source_rows": file_rows, "aggregates": aggregates,
            "paired_vs_uncompressed": contrasts, "noise_contrast_changes": changes,
            "evaluation_level_comparisons": level_contrasts, "paired_summary": comparison_summary,
            "overlap": overlap, "statistics": {"interval_type": "observed minimum-to-maximum across preset runs, descriptive only",
            "uncertainty_display": "sample standard deviation across preset runs; not a standard error or confidence interval",
            "hypothesis_tests_performed": False, "population_confidence_intervals_reported": False,
            "reason": "Small preset run sets reuse source and bearing identities and change split/operator/training together; no justified IID physical replication unit for population inference",
            "multiple_comparison_policy": "all locked contrasts disclosed; no hypothesis testing, no significant-winner selection",
            "effect_size": "paired absolute Macro-F1/recall differences, expressed in percentage points for reporting"}}


def make_figures(data: dict, root: Path) -> list[dict]:
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mpl_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False,
                               "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    catalog = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    for i, dataset in enumerate(("pu", "cwru")):
        for j, metric in enumerate(("window_macro_f1", "source_macro_f1")):
            ax = axes[i, j]
            rows = [r for r in data["paired_vs_uncompressed"] if r["dataset"] == dataset and r["method"] == "pca"]
            seeds = sorted({r["seed"] for r in rows})
            matrix = np.asarray([[next(r[metric + "_delta"] for r in rows if r["seed"] == seed and r["condition"] == condition) * 100 for condition in CONDITIONS] for seed in seeds])
            for values in matrix:
                ax.plot(range(3), values, marker="o", ms=3, lw=0.7, color="#777777", alpha=0.45)
            ax.errorbar(range(3), matrix.mean(axis=0), yerr=matrix.std(axis=0, ddof=1), color="#D55E00", lw=1.5, marker="D", ms=4, capsize=3)
            ax.axhline(0, color="#222222", lw=0.7, linestyle="--")
            ax.set_xticks(range(3), ["Original", "20 dB", "10 dB"])
            ax.set_ylabel("PCA − uncompressed (pp)")
            ax.set_title(f"{dataset.upper()}  {'Window' if j == 0 else 'Source-file'} Macro-F1  (n={len(seeds)} runs)", fontsize=10)
            ax.grid(axis="y", alpha=0.18)
    name = folder / "figure_01_pca_paired_contrasts"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(name.with_suffix("." + extension), dpi=600)
    plt.close(fig)
    catalog.append({"file": str(name.with_suffix(".png")), "purpose": "Distinguish PCA superiority under noise from strict clean-to-noise ranking reversal",
                    "caption": "PCA-minus-uncompressed Macro-F1 on PU and CWRU. Grey lines show paired preset runs; orange points show means and error bars show sample standard deviations. Original denotes no added experimental noise. Runs reuse data identities and are not independent physical replications."})
    colors = {"no_cs": "#222222", "ilsomp": "#0072B2", "downsample": "#009E73", "pca": "#D55E00", "random_projection": "#CC79A7"}
    markers = {"no_cs": "o", "ilsomp": "s", "downsample": "^", "pca": "D", "random_projection": "v"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for i, dataset in enumerate(("pu", "cwru")):
        for j, metric in enumerate(("window_macro_f1", "source_macro_f1")):
            ax = axes[i, j]
            for method in METHODS:
                group = [next(r for r in data["aggregates"] if (r["dataset"], r["method"], r["condition"]) == (dataset, method, condition)) for condition in CONDITIONS]
                means = np.asarray([r["metrics"][metric]["mean"] for r in group])
                sd = np.asarray([r["metrics"][metric]["sample_sd"] for r in group])
                ax.plot(range(3), means, color=colors[method], marker=markers[method], ms=3.5, lw=1.3, label=LABELS[method])
                ax.fill_between(range(3), np.maximum(0, means - sd), np.minimum(1, means + sd), color=colors[method], alpha=0.08, linewidth=0)
            ax.set_ylim(0, 1.03)
            ax.set_xticks(range(3), ["Original", "20 dB", "10 dB"])
            ax.set_ylabel("Macro-F1")
            ax.set_title(f"{dataset.upper()}  {'Window' if j == 0 else 'Source-file'}", fontsize=10)
            ax.grid(axis="y", alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    name = folder / "figure_02_all_method_noise_profiles"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(name.with_suffix("." + extension), dpi=600)
    plt.close(fig)
    catalog.append({"file": str(name.with_suffix(".png")), "purpose": "Show all five fixed pipelines rather than selecting a favourable method or dataset",
                    "caption": "Window and source-file Macro-F1 of all five pipelines under original observations and 20/10 dB added Gaussian noise. Lines show preset-run means; bands show mean ± sample standard deviation, clipped to the valid score interval. No models or PCA transformers were refitted."})
    return catalog


def reports(data: dict, catalog: list[dict], root: Path) -> None:
    lines = ["# 两数据集固定模型加噪扩展分析", "", "全部指定单元完成后汇总；没有根据结果选择种子、噪声或方法。", "",
             "## 五种方法的完整结果", "", "数值为预设运行的均值±样本标准差，非标准误或总体置信区间。PU为3次、CWRU为5次运行。", "",
             "| 数据集 | 方法 | 条件 | 窗口Macro-F1 | 源文件Macro-F1 | 误判文件均数 |", "|---|---|---|---:|---:|---:|"]
    for row in data["aggregates"]:
        m = row["metrics"]
        lines.append(f"| {row['dataset'].upper()} | {LABELS[row['method']]} | {row['condition']} | {m['window_macro_f1']['mean']:.4f}±{m['window_macro_f1']['sample_sd']:.4f} | {m['source_macro_f1']['mean']:.4f}±{m['source_macro_f1']['sample_sd']:.4f} | {m['failed_source_count']['mean']:.2f} |")
    lines.extend(["", "## PCA与不压缩的配对比较", "", "严格反转要求同一运行原观测差值<0且加噪差值>0；原观测本已为正不计为反转。", ""])
    for dataset in ("pu", "cwru"):
        lines.extend([f"### {dataset.upper()}", ""])
        for condition in CONDITIONS:
            row = next(r for r in data["paired_summary"] if (r["dataset"], r["method"], r["condition"]) == (dataset, "pca", condition))
            stat = row["metrics"]["window_macro_f1"]
            message = f"- {condition}：PCA−不压缩窗口Macro-F1 = {stat['mean']*100:+.2f}±{stat['sample_sd']*100:.2f}个百分点；逐运行正/负/零 = {stat['positive_runs']}/{stat['negative_runs']}/{stat['zero_runs']}。"
            if condition != "original":
                change = row["contrast_changes"]["window_macro_f1"]
                message += f" 相对原观测的差值变化为{change['mean']*100:+.2f}个百分点；严格负转正{row['strict_window_reversal_runs']}/{stat['n']}次。"
            lines.append(message)
    lines.extend(["", "## 可写与不可写的结论", "", "### 可写", ""])
    for dataset in ("pu", "cwru"):
        original = next(r for r in data["paired_summary"] if (r["dataset"], r["method"], r["condition"]) == (dataset, "pca", "original"))
        noisy = next(r for r in data["paired_summary"] if (r["dataset"], r["method"], r["condition"]) == (dataset, "pca", "10dB"))
        first, last = original["metrics"]["window_macro_f1"]["mean"], noisy["metrics"]["window_macro_f1"]["mean"]
        lines.append(f"- {dataset.upper()}：汇总PCA相对不压缩窗口差值由{first*100:+.2f}变为{last*100:+.2f}个百分点；数据集均值层面的负转正 = {first < 0 and last > 0}。逐运行情况见上表与CSV。")
    lines.extend(["", "### 不可写", "", "- 不能把新增条件测试称为新的独立测试集确认。", "- 不能把预设运行当作独立设备、实验室或物理轴承复制。", "- 不能将源文件判决直接改称物理轴承级可靠性。", "- 不能仅由PCA反超推断已经证明降噪机制、现场部署收益或真实采集能耗优势。", "- 不能将少数运行或数据集中的反转写成所有条件下均成立。", "", "## 输入重叠", ""])
    for dataset, record in data["overlap"].items():
        lines.append(f"- {dataset.upper()}：{record['source_file_occurrences']}次文件出现对应{record['unique_source_paths_across_runs']}条唯一源路径；重复出现{record['repeated_source_occurrences']}次。")
    lines.extend(["", "## 图件解释", ""])
    for entry in catalog:
        lines.extend([f"### {Path(entry['file']).name}", "", entry["purpose"], "", entry["caption"], ""])
    (root / "analysis-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    appendix = ["# 统计说明", "", "## 单位与配对", "", "同一数据集、种子、模型检查点与源文件集合内作原观测/加噪配对，方法间输入一致。窗口是文件内嵌套观测，源文件和轴承身份可在种子之间重叠。", "", "## 描述性效应量", "", "完整报告各运行Macro-F1和召回差值、均值、样本标准差、中位数、最小值、最大值及方向计数；区间为观测到的运行间范围，不是95%置信区间。", "", "## 推断选择", "", "没有以3/5次预设运行做总体t检验、Wilcoxon检验、显著性标记或总体置信区间。理由是独立物理抽样单位未获支持，且原测试已经用于发现。不能通过窗口级重采样虚增样本量，也不能以小样本正态性检验未拒绝来证明独立性。", "", "## 多重比较", "", "主比较为PCA相对不压缩的窗口Macro-F1及其噪声条件差值变化；其他方法、源文件指标和类别召回为同时披露的次要描述。没有p值筛选，也没有选择最佳seed或只保留有利SNR。", "", "## 判定规则", "", "同一运行的原观测差值<0且加噪差值>0才计严格负转正。均值排序与逐运行排序分开报告，二者不互相替代。"]
    (root / "stats-appendix.md").write_text("\n".join(appendix) + "\n", encoding="utf-8")
    (root / "figure-catalog.md").write_text("\n\n".join("# " + Path(c["file"]).stem + "\n\nPurpose: " + c["purpose"] + "\n\nCaption: " + c["caption"] + "\n\nInterpretation: Compare with the paired numerical tables; distinguish aggregate and individual-run reversals. Do not infer a denoising mechanism or physical-bearing independence." for c in catalog) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    path = args.lock.resolve()
    validation = validate(path)
    lock = read_json(path)
    summary_path = path.parent / "noise_robustness_extension.json"
    summary = read_json(summary_path)
    root = args.output_root.resolve() if args.output_root else path.parent / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    data = build_analysis(summary, lock["protocol"])
    data["signature"] = lock["signature"]
    data["inputs"] = [artifact(path), artifact(summary_path), artifact(Path(__file__))]
    data["validation"] = validation
    save_json(root / "analysis.json", data)
    for name, rows in (("metrics", data["rows"]), ("class_recalls", data["class_recalls"]), ("source_rows", data["source_rows"]),
                       ("paired_vs_uncompressed", data["paired_vs_uncompressed"]), ("noise_contrast_changes", data["noise_contrast_changes"]),
                       ("evaluation_level_comparisons", data["evaluation_level_comparisons"])):
        write_csv(root / (name + ".csv"), rows)
    catalog = make_figures(data, root)
    reports(data, catalog, root)
    save_json(root / "analysis_manifest.json", {"signature": lock["signature"], "training_performed": False,
              "inference_performed": False, "raw_signal_values_opened": False, "independent_confirmation": False,
              "files": [artifact(p) for p in sorted(root.rglob("*")) if p.is_file() and ".mpl_cache" not in p.parts and p.name != "analysis_manifest.json"]})
    print(json.dumps({"complete": True, "analysis": str(root / "analysis.json"), "report": str(root / "analysis-report.md")}, indent=2))


if __name__ == "__main__":
    main()
