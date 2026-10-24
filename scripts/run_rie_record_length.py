from __future__ import annotations

import argparse
import json
from pathlib import Path

from rie_supplement_support import aggregate, align, arrays, artifact, build_protocol, evaluate, load_lock, ordered_records, prepare, read_json, save_json


def run(lock_path: Path) -> None:
    lock = load_lock(lock_path, "record_length")
    root = lock_path.resolve().parent
    cells = []
    for item in lock["protocol"]["runs"]:
        reference = None
        for method, entry in item["methods"].items():
            payload = arrays(entry["prediction"]["path"])
            if reference is not None:
                align(reference, payload)
            reference = payload
            records = ordered_records(read_json(entry["manifest"]["path"]), payload)
            path = root / "cells" / f"{item['dataset']}_seed{item['seed']}_{method}.json"
            if path.exists():
                stored = read_json(path)
                if stored["signature"] != lock["signature"] or stored["input_sha256"] != entry["prediction"]["sha256"]:
                    raise ValueError("Completed cell belongs to a different protocol")
            else:
                rows = []
                for length in item["window_lengths"]:
                    rows.append({"dataset": item["dataset"], "seed": item["seed"], "method": method,
                                 "length": length, "observation_seconds": None if length == "full" else length * 1024 / 12000,
                                 "metrics": evaluate(payload, records, length)})
                stored = {"signature": lock["signature"], "input_sha256": entry["prediction"]["sha256"], "cells": rows}
                save_json(path, stored)
            cells.extend(stored["cells"])
            print(f"[{item['dataset']} seed{item['seed']} {method}] complete", flush=True)
    references = {(c["dataset"], c["seed"], c["length"]): c for c in cells if c["method"] == "no_cs"}
    paired = []
    for cell in cells:
        if cell["method"] == "no_cs":
            continue
        ref = references[(cell["dataset"], cell["seed"], cell["length"])]
        paired.append({key: cell[key] for key in ("dataset", "seed", "method", "length")} | {
            "source_macro_f1_delta_vs_no_cs": cell["metrics"]["source"]["macro_f1"] - ref["metrics"]["source"]["macro_f1"],
            "failed_source_count_delta_vs_no_cs": cell["metrics"]["failed_source_count"] - ref["metrics"]["failed_source_count"]})
    summary = {"complete": True, "signature": lock["signature"], "protocol_lock": artifact(lock_path),
               "training_performed": False, "inference_performed": False, "uored_test_opened": False,
               "new_independent_test_opened": False,
               "cells": cells, "paired_comparisons": paired,
               "aggregate": aggregate(cells, ["dataset", "method", "length"]),
               "interpretation": "chronological prefix voting; all lengths use the same source cohort; seeds are descriptive repeats, not new physical replications"}
    save_json(root / "record_length_summary.json", summary)
    print(json.dumps({"complete": True, "summary": str(root / "record_length_summary.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["prepare", "run"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-signature")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("F:/paper1/artifacts_v2/reports/rie_record_length"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(build_protocol("record_length", ["pu", "cwru"]), args.dry_run, args.confirm_signature, args.output_root)
    else:
        if args.lock is None:
            parser.error("run requires --lock")
        run(args.lock)


if __name__ == "__main__":
    main()
