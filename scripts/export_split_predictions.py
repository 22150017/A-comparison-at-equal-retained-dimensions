





from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.attention_resnet import build_backbone
from src.training.train_classifier import _prepare_images, image_dataloaders
from src.utils.io_tools import export_json
from src.utils.reproducibility import canonical_hash, file_sha256, seed_everything


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _collect(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    fast_training: bool,
    channels_last: bool,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        report_every = max(1, len(loader) // 5)
        started = time.perf_counter()
        completed = 0
        for batch_index, (images, labels) in enumerate(loader, 1):
            images = _prepare_images(images, device, fast_training, channels_last)
            logits_parts.append(model(images).float().cpu().numpy())
            labels_parts.append(labels.numpy())
            completed += len(labels)
            if batch_index % report_every == 0 or batch_index == len(loader):
                elapsed = max(time.perf_counter() - started, 1e-9)
                speed = completed / elapsed
                remaining = max(len(loader.dataset) - completed, 0)
                print(
                    f"[{progress_label}] {completed}/{len(loader.dataset)} samples, "
                    f"speed={speed:.1f} samples/s, ETA={remaining / speed:.0f}s"
                )
    if not logits_parts:
        raise ValueError("cannot export predictions for an empty split")
    return np.concatenate(logits_parts), np.concatenate(labels_parts)


def _sample_ids(loader) -> np.ndarray:
    values = [Path(path).stem for path, _ in loader.dataset.samples]
    if len(values) != len(set(values)):
        raise ValueError("split contains duplicate sample IDs")
    width = max(len(value) for value in values)
    return np.asarray(values, dtype=f"<U{width}")


def export_split_predictions(
    data_root: Path,
    result_root: Path,
    backbone: str = "resnet34_ema",
    batch_size: int = 64,
    eval_num_workers: int = 0,
    force: bool = False,
    splits: tuple[str, ...] = ("val", "test"),
) -> Path:
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"val", "test"}:
        raise ValueError("splits must be a nonempty unique subset of val/test")
    manifest_path = data_root / "dataset_manifest.json"
    metrics_path = result_root / f"{backbone}_metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("dataset manifest and trained metrics JSON are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    dataset_signature = str(manifest.get("signature"))
    if metrics.get("run_config", {}).get("dataset_signature") != dataset_signature:
        raise ValueError("checkpoint metrics do not belong to the supplied dataset root")
    run_signature = str(metrics.get("run_signature", ""))
    if not run_signature:
        raise ValueError("metrics JSON has no run signature")
    checkpoint_file = str(metrics.get("checkpoint_file", f"{backbone}_best.pt"))
    checkpoint = result_root / checkpoint_file
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    fast_training = bool(metrics.get("run_config", {}).get("performance", {}).get("fast_training"))
    seed = int(metrics.get("run_config", {}).get("seed", 42))
    export_config = {
        "dataset_signature": dataset_signature,
        "run_signature": run_signature,
        "checkpoint_sha256": file_sha256(checkpoint),
        "backbone": backbone,
        "batch_size": batch_size,
        "eval_num_workers": eval_num_workers,
        "fast_training": fast_training,
    }
    export_signature = canonical_hash(export_config)
    
    scope_suffix = "" if splits == ("val", "test") else "_" + "_".join(splits)
    if scope_suffix:
        export_config["splits"] = list(splits)
        export_signature = canonical_hash(export_config)
    metadata_path = result_root / f"{backbone}_split_predictions{scope_suffix}_metadata.json"
    if metadata_path.exists() and not force:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("export_signature") != export_signature:
            raise RuntimeError(
                "existing prediction sidecar has a different configuration; use --force explicitly"
            )
        expected = [result_root / previous["files"][split]["name"] for split in splits]
        if all(path.exists() and file_sha256(path) == previous["files"][split]["sha256"] for split, path in zip(splits, expected)):
            return metadata_path

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader, num_classes, class_names = image_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
        fast_training=fast_training,
        prefetch_factor=2,
        eval_num_workers=eval_num_workers,
        include_test="test" in splits,
    )
    model = build_backbone(backbone, num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    channels_last = bool(fast_training and device.type == "cuda")
    if channels_last:
        model.to(memory_format=torch.channels_last)

    files: dict[str, dict] = {}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        if split not in splits:
            continue
        logits, labels = _collect(
            model, loader, device, fast_training, channels_last, f"export {split} predictions"
        )
        sample_ids = _sample_ids(loader)
        if len(sample_ids) != len(labels):
            raise RuntimeError(f"{split} sample IDs and predictions have different lengths")
        target_indices = np.asarray([target for _, target in loader.dataset.samples], dtype=np.int64)
        if not np.array_equal(labels, target_indices):
            raise RuntimeError(f"{split} DataLoader order differs from ImageFolder sample order")
        path = result_root / f"{backbone}{scope_suffix}_{split}_predictions.npz"
        _atomic_npz(
            path,
            sample_ids=sample_ids,
            y_true=labels.astype(np.int64),
            logits=logits.astype(np.float32),
            class_names=np.asarray(class_names),
        )
        files[split] = {
            "name": path.name,
            "sha256": file_sha256(path),
            "sample_count": int(len(labels)),
        }

    metadata = {
        "export_signature": export_signature,
        "export_config": export_config,
        "class_names": class_names,
        "sample_identity": "PNG filename stem copied from dataset_manifest.sample_id",
        "order": "torchvision ImageFolder lexical order; alignment must still use sample_id",
        "files": files,
    }
    export_json(metadata_path, metadata)
    return metadata_path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset-root", type=Path, required=True)
    cli.add_argument("--result-root", type=Path, required=True)
    cli.add_argument("--backbone", default="resnet34_ema")
    cli.add_argument("--batch-size", type=int, default=64)
    cli.add_argument("--eval-num-workers", type=int, default=0)
    cli.add_argument("--force", action="store_true")
    return cli


def main() -> None:
    args = parser().parse_args()
    output = export_split_predictions(
        args.dataset_root,
        args.result_root,
        backbone=args.backbone,
        batch_size=args.batch_size,
        eval_num_workers=args.eval_num_workers,
        force=args.force,
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
