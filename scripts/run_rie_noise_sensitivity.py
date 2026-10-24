from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rie_supplement_support import aggregate, align, arrays, artifact, build_protocol, evaluate, load_lock, ordered_records, prepare, read_json, save_json, save_npz


def noisy_window(signal: np.ndarray, sample_id: str, snr: float, master_seed: int) -> tuple[np.ndarray, float, str]:
    seed = int.from_bytes(hashlib.sha256(f"{master_seed}:{sample_id}".encode()).digest()[:8], "little")
    noise = np.random.default_rng(seed).standard_normal(len(signal))
    noise -= noise.mean()
    energy = float(np.dot(signal, signal))
    if energy <= 0:
        raise ValueError(f"Zero-energy observed window: {sample_id}")
    noise *= np.sqrt(energy / (10 ** (snr / 10) * np.dot(noise, noise)))
    perturbed = np.asarray(signal + noise, dtype="<f8")
    actual = float(10 * np.log10(energy / np.dot(noise, noise)))
    return perturbed, actual, hashlib.sha256(perturbed.tobytes()).hexdigest()


def materialize(item: dict, records: list[dict]) -> np.ndarray:
    from src.data.cwru_loader import load_cwru_mat_file
    from src.data.pu_loader import extract_vibration, resample_signal
    corrected_sources = None
    if item["dataset"] == "cwru":
        manifest = read_json(item["methods"]["no_cs"]["manifest"]["path"])
        inventory = manifest["condition"]["signal"].get("cwru_source_inventory")
        if inventory is not None:
            corrected_sources = {
                str(Path(source["path"]).resolve()): row
                for row in inventory["sources"]
                for source in manifest["condition"]["sources"]
                if Path(source["path"]).as_posix().endswith(row["relative_path"])
            }
    result = np.empty((len(records), 1024), dtype=np.float64)
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["source_file"]].append((index, int(record["segment_id"])))
    for number, (source, indices) in enumerate(sorted(grouped.items()), 1):
        path = Path(source)
        if item["dataset"] == "pu":
            vector = resample_signal(extract_vibration(path), 64000, 12000)
        elif corrected_sources is not None:
            from src.data.cwru_corrected import read_source

            row = corrected_sources[str(path.resolve())]
            source_root = path.resolve()
            for _ in Path(row["relative_path"]).parts:
                source_root = source_root.parent
            _, vector = read_source(source_root, row)
        else:
            vector, _ = load_cwru_mat_file(path, prefer_sensor="DE")
        for index, segment in indices:
            window = vector[segment * 1024:(segment + 1) * 1024]
            if len(window) != 1024:
                raise ValueError(f"Invalid formal window position: {source}/{segment}")
            result[index] = window
        if number % 32 == 0 or number == len(grouped):
            print(f"[raw {item['dataset']} seed{item['seed']}] {number}/{len(grouped)} sources", flush=True)
    return result


def evaluate_cell(protocol: dict, item: dict, method: str, snr: float, root: Path,
                  lock_signature: str, raw: np.ndarray, records: list[dict], clean: dict) -> dict:
    import torch
    from PIL import Image
    from scripts.benchmark_followup_efficiency import _method_transform
    from scripts.run_clean_study import _stft_image
    from src.data.cwru_loader import normalize_signal
    from src.models.attention_resnet import build_backbone
    from src.reconstruction.compression_baselines import load_pca_baseline
    from src.training.train_classifier import _fast_image_transform, _image_transform, _prepare_images
    from src.utils.reproducibility import seed_everything

    entry = item["methods"][method]
    cell_id = f"{item['dataset']}_seed{item['seed']}_{method}_snr{snr:g}"
    directory = root / "cells" / cell_id
    completed = directory / "cell.json"
    if completed.exists():
        cell = read_json(completed)
        if cell["signature"] != lock_signature:
            raise ValueError("Completed cell signature differs")
        artifact(cell["prediction"]["path"], cell["prediction"]["sha256"])
        print(f"[{cell_id}] complete; skipping", flush=True)
        return cell
    seed_everything(item["seed"])
    pca = load_pca_baseline(Path(entry["pca"]["path"])) if method == "pca" else None
    transform = _method_transform(method, 0.5, item["seed"], pca)
    check_rows = []
    for check in entry["clean_image_checks"]:
        index = check["index"]
        generated = np.asarray(_stft_image(transform(normalize_signal(raw[index], "max_abs")), 224, 64, 32))
        with Image.open(check["path"]) as image:
            existing = np.asarray(image.convert("RGB"))
        if not np.array_equal(generated, existing):
            raise ValueError(f"Clean frontend no longer reproduces formal pixels: {cell_id}/{clean['sample_ids'][index]}")
        check_rows.append({"sample_id": str(clean["sample_ids"][index]), "pixels_equal": True})
    export_cfg = read_json(entry["prediction_metadata"]["path"])["export_config"]
    fast = bool(export_cfg["fast_training"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels_last = fast and device.type == "cuda"
    model = build_backbone(export_cfg["backbone"], num_classes=len(clean["class_names"]), pretrained=False)
    model.load_state_dict(torch.load(entry["checkpoint"]["path"], map_location="cpu", weights_only=True))
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()
    image_transform = _fast_image_transform() if fast else _image_transform()
    chunks, timing = [], {"frontend_seconds": 0.0, "inference_seconds": 0.0}
    chunk_size, batch_size = protocol["chunk_size"], protocol["batch_size"]
    total = len(raw)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        shard_path = directory / "shards" / f"{start:07d}_{end:07d}.npz"
        sidecar_path = shard_path.with_suffix(".json")
        if shard_path.exists() and sidecar_path.exists():
            sidecar = read_json(sidecar_path)
            if sidecar["signature"] != lock_signature or sidecar["cell_id"] != cell_id:
                raise ValueError("Resumed shard differs from this protocol")
            artifact(shard_path, sidecar["sha256"])
            shard = arrays(shard_path)
        else:
            started = time.perf_counter()
            images, measured, hashes = [], [], []
            for i in range(start, end):
                noisy, actual, sha = noisy_window(raw[i], str(clean["sample_ids"][i]), snr, protocol["noise_master_seed"])
                images.append(image_transform(_stft_image(transform(normalize_signal(noisy, "max_abs")), 224, 64, 32)))
                measured.append(actual)
                hashes.append(sha)
            frontend_seconds = time.perf_counter() - started
            logits = []
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                for offset in range(0, len(images), batch_size):
                    batch = _prepare_images(torch.stack(images[offset:offset + batch_size]), device, fast, channels_last)
                    logits.append(model(batch).float().cpu().numpy())
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - started
            shard = {"sample_ids": clean["sample_ids"][start:end], "y_true": clean["y_true"][start:end],
                     "class_names": clean["class_names"], "logits": np.concatenate(logits),
                     "actual_added_snr_db": np.asarray(measured), "noisy_input_sha256": np.asarray(hashes)}
            save_npz(shard_path, **shard)
            sidecar = {"signature": lock_signature, "cell_id": cell_id, "sha256": artifact(shard_path)["sha256"],
                       "frontend_seconds": frontend_seconds, "inference_seconds": inference_seconds}
            save_json(sidecar_path, sidecar)
        for key in ("sample_ids", "y_true"):
            if not np.array_equal(shard[key], clean[key][start:end]):
                raise ValueError("Resumed shard order differs from formal predictions")
        if not np.array_equal(shard["class_names"], clean["class_names"]):
            raise ValueError("Resumed shard class order differs")
        chunks.append(shard)
        for key in timing:
            timing[key] += sidecar[key]
        elapsed_processing = timing["frontend_seconds"] + timing["inference_seconds"]
        eta = elapsed_processing * (total - end) / end
        print(f"[{cell_id}] {end}/{total} windows; processing ETA={eta:.0f}s", flush=True)
    prediction = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in
                  ("sample_ids", "y_true", "logits", "actual_added_snr_db", "noisy_input_sha256")}
    prediction["class_names"] = clean["class_names"]
    align(clean, prediction)
    prediction_path = directory / "test_predictions.npz"
    save_npz(prediction_path, **prediction)
    metrics, reference = evaluate(prediction, records), evaluate(clean, records)
    delta = {"window_macro_f1": metrics["window"]["macro_f1"] - reference["window"]["macro_f1"],
             "minimum_class_recall": metrics["window"]["minimum_class_recall"] - reference["window"]["minimum_class_recall"],
             "source_macro_f1": metrics["source"]["macro_f1"] - reference["source"]["macro_f1"],
             "failed_source_count": metrics["failed_source_count"] - reference["failed_source_count"],
             "per_class_recall": {key: value - reference["window"]["per_class_recall"][key]
                                  for key, value in metrics["window"]["per_class_recall"].items()}}
    cell = {"signature": lock_signature, "dataset": item["dataset"], "seed": item["seed"], "method": method,
            "snr_db": snr, "prediction": artifact(prediction_path), "metrics": metrics, "clean_reference": reference,
            "delta_vs_clean": delta, "timing": timing, "clean_frontend_checks": check_rows,
            "checkpoint": entry["checkpoint"], "runtime": {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
            "training_performed": False, "pca_refitted": False, "uored_test_opened": False}
    save_json(completed, cell)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cell


def run(lock_path: Path) -> None:
    lock = load_lock(lock_path, "noise_sensitivity")
    protocol, root, cells = lock["protocol"], lock_path.resolve().parent, []
    for item in protocol["runs"]:
        baseline = arrays(item["methods"]["no_cs"]["prediction"]["path"])
        records = ordered_records(read_json(item["methods"]["no_cs"]["manifest"]["path"]), baseline)
        raw = None
        for method, entry in item["methods"].items():
            clean = arrays(entry["prediction"]["path"])
            align(baseline, clean)
            for snr in protocol["snr_db"]:
                completed = root / "cells" / f"{item['dataset']}_seed{item['seed']}_{method}_snr{snr:g}" / "cell.json"
                if completed.exists():
                    cell = read_json(completed)
                    if cell["signature"] != lock["signature"]:
                        raise ValueError("Completed cell belongs to a different protocol")
                    artifact(cell["prediction"]["path"], cell["prediction"]["sha256"])
                    cells.append(cell)
                    print(f"[{completed.parent.name}] complete; skipping", flush=True)
                    continue
                if raw is None:
                    raw = materialize(item, records)
                cells.append(evaluate_cell(protocol, item, method, snr, root, lock["signature"], raw, records, clean))
    summary = {"complete": True, "signature": lock["signature"], "protocol_lock": artifact(lock_path),
               "training_performed": False, "inference_performed": True, "pca_refitted": False,
               "uored_test_opened": False, "new_independent_test_opened": False, "cells": cells,
               "aggregate": aggregate(cells, ["dataset", "method", "snr_db"]),
               "interpretation": "added-noise sensitivity on existing formal test windows; no classifier training or PCA refitting"}
    save_json(root / "noise_sensitivity_summary.json", summary)
    print(json.dumps({"complete": True, "summary": str(root / "noise_sensitivity_summary.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["prepare", "run"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-signature")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--dataset", choices=["pu", "cwru"], default="pu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output-root", type=Path, default=Path("F:/paper1/artifacts_v2/studies/rie_noise_sensitivity"))
    args = parser.parse_args()
    if args.stage == "prepare":
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("duplicate seeds are not allowed")
        protocol = build_protocol("noise_sensitivity", [args.dataset], args.seeds, args.batch_size, args.chunk_size)
        prepare(protocol, args.dry_run, args.confirm_signature, args.output_root)
    else:
        if args.lock is None:
            parser.error("run requires --lock")
        run(args.lock)


if __name__ == "__main__":
    main()
