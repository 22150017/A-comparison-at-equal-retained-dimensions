from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from src.models.attention_resnet import build_backbone
from src.utils.io_tools import export_json, plot_confusion_matrix, plot_training_curves
from src.utils.reproducibility import (
    canonical_hash, capture_rng_state, file_sha256, restore_rng_state,
    runtime_metadata, seed_everything,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_NORMALIZATION_CACHE: dict[
    tuple[str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _fast_image_transform() -> transforms.Compose:
    
    return transforms.Compose([transforms.PILToTensor()])


def _prepare_images(
    images: torch.Tensor,
    device: torch.device,
    fast_training: bool = False,
    channels_last: bool = False,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if fast_training:
        images = images.float().mul_(1.0 / 255.0)
        if images.shape[-2:] != (224, 224):
            images = torch.nn.functional.interpolate(
                images,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        cache_key = (device.type, device.index, images.dtype)
        if cache_key not in _NORMALIZATION_CACHE:
            _NORMALIZATION_CACHE[cache_key] = (
                images.new_tensor(IMAGENET_MEAN).view(1, -1, 1, 1),
                images.new_tensor(IMAGENET_STD).view(1, -1, 1, 1),
            )
        mean, std = _NORMALIZATION_CACHE[cache_key]
        images.sub_(mean).div_(std)
    if channels_last and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def image_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    fast_training: bool = False,
    prefetch_factor: int = 4,
    eval_num_workers: int = 0,
    balanced_sampling: str = "none",
    include_test: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None, int, list[str]]:
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if eval_num_workers < 0:
        raise ValueError("eval_num_workers cannot be negative")
    if balanced_sampling not in {"none", "class-balanced"}:
        raise ValueError("balanced_sampling must be 'none' or 'class-balanced'")
    transform = _fast_image_transform() if fast_training else _image_transform()
    train_dataset = datasets.ImageFolder(str(data_root / "train"), transform=transform)
    class_names = list(train_dataset.classes)
    class_to_idx = dict(train_dataset.class_to_idx)
    datasets_by_split = {"train": train_dataset}
    evaluation_splits = ("val", "test") if include_test else ("val",)
    for split in evaluation_splits:
        dataset = datasets.ImageFolder(str(data_root / split), transform=transform)
        unknown = sorted(set(dataset.classes) - set(class_names))
        if unknown:
            raise ValueError(f"unknown classes in {split}: {unknown}")
        local_to_global = {
            local_index: class_to_idx[class_name]
            for class_name, local_index in dataset.class_to_idx.items()
        }
        dataset.samples = [
            (path, local_to_global[target]) for path, target in dataset.samples
        ]
        dataset.imgs = dataset.samples
        dataset.targets = [target for _, target in dataset.samples]
        dataset.classes = class_names
        dataset.class_to_idx = class_to_idx
        datasets_by_split[split] = dataset
    for split, dataset in datasets_by_split.items():
        if not dataset.samples:
            raise ValueError(f"empty dataset split: {data_root / split}")
    generator = torch.Generator().manual_seed(seed)
    loader_options: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor if fast_training else 2
    if balanced_sampling == "class-balanced":
        class_weights = balanced_class_weights(datasets_by_split["train"])
        targets = torch.as_tensor(datasets_by_split["train"].targets, dtype=torch.long)
        train_sampler = WeightedRandomSampler(
            weights=class_weights[targets],
            num_samples=len(datasets_by_split["train"]),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            datasets_by_split["train"],
            shuffle=False,
            sampler=train_sampler,
            **loader_options,
        )
    else:
        train_loader = DataLoader(
            datasets_by_split["train"],
            shuffle=True,
            generator=generator,
            **loader_options,
        )
    if fast_training:
        eval_options: dict = {
            "batch_size": batch_size,
            "num_workers": eval_num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": False,
        }
        if eval_num_workers > 0:
            eval_options["prefetch_factor"] = min(prefetch_factor, 2)
        val_loader = DataLoader(datasets_by_split["val"], shuffle=False, **eval_options)
        test_loader = (
            DataLoader(datasets_by_split["test"], shuffle=False, **eval_options)
            if include_test
            else None
        )
    else:
        val_loader = DataLoader(datasets_by_split["val"], shuffle=False, **loader_options)
        test_loader = (
            DataLoader(datasets_by_split["test"], shuffle=False, **loader_options)
            if include_test
            else None
        )
    return train_loader, val_loader, test_loader, len(class_names), class_names


def freeze_partial_backbone(model: nn.Module, freeze_ratio: float = 0.5) -> None:
    params = list(model.parameters())
    cutoff = int(len(params) * freeze_ratio)
    for param in params[:cutoff]:
        param.requires_grad = False


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_label: str | None = None,
    fast_training: bool = False,
    channels_last: bool = False,
    class_names: list[str] | None = None,
) -> Dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    started = time.perf_counter()
    report_every = max(1, len(loader) // 5)
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, 1):
            images = _prepare_images(images, device, fast_training, channels_last)
            logits = model(images)
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())
            if progress_label and (batch_index % report_every == 0 or batch_index == len(loader)):
                elapsed = max(time.perf_counter() - started, 1e-9)
                speed = len(y_true) / elapsed
                remaining = max(len(loader.dataset) - len(y_true), 0)
                eta = remaining / speed if speed else None
                print(
                    f"[{progress_label}] {len(y_true)}/{len(loader.dataset)} samples, "
                    f"speed={speed:.1f} samples/s, ETA={_format_duration(eta)}"
                )
    if not y_true:
        raise ValueError("cannot evaluate an empty loader")
    metric_labels = list(range(len(class_names))) if class_names is not None else None
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=metric_labels, average="macro", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, labels=metric_labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=metric_labels, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=metric_labels),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=metric_labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "sample_count": len(y_true),
    }


def _dataset_signature(data_root: Path) -> str:
    manifest = data_root / "dataset_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing required dataset manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return str(payload.get("signature") or canonical_hash(payload))


def _metrics_from_json(payload: dict) -> dict:
    for key in ("metrics", "paper_metrics", "train_metrics", "val_metrics", "test_metrics"):
        if isinstance(payload.get(key), dict) and isinstance(payload[key].get("confusion_matrix"), list):
            payload[key]["confusion_matrix"] = np.asarray(payload[key]["confusion_matrix"])
    return payload


def balanced_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    
    targets = torch.as_tensor(dataset.targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=len(dataset.classes)).float()
    if torch.any(counts <= 0):
        raise ValueError(f"cannot balance an empty training class: {counts.tolist()}")
    return counts.sum() / (len(dataset.classes) * counts)


def _atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sampler_generator(loader: DataLoader) -> torch.Generator | None:
    generator = getattr(loader.sampler, "generator", None)
    return generator if isinstance(generator, torch.Generator) else None


def _capture_training_rng_state(loader: DataLoader) -> dict:
    generator = _sampler_generator(loader)
    return {
        "global": capture_rng_state(),
        "sampler": generator.get_state() if generator is not None else None,
    }


def _restore_training_rng_state(state: dict, train_loader: DataLoader, val_loader: DataLoader, seed: int) -> None:
    generator = _sampler_generator(train_loader)
    saved_sampler = state["sampler"]
    if saved_sampler is not None:
        if generator is None:
            raise RuntimeError("checkpoint sampler state has no matching training sampler")
        generator.set_state(saved_sampler.detach().cpu())
    if train_loader.persistent_workers:
        train_loader.generator = torch.Generator().manual_seed(seed)
    if val_loader.persistent_workers:
        val_loader.generator = torch.Generator().manual_seed(seed + 1)
    restore_rng_state(state["global"])


def train_classifier(
    data_root: Path,
    export_root: Path,
    backbone: str = "resnet34",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    freeze_ratio: float = 0.5,
    num_workers: int = 0,
    seed: int = 42,
    amp: bool = False,
    fast_training: bool = False,
    prefetch_factor: int = 4,
    eval_num_workers: int = 0,
    class_weighting: str = "none",
    balanced_sampling: str = "none",
    report_train_metrics: bool = False,
    checkpoint_selection: str = "best-val",
    evaluation_scope: str = "formal-test",
) -> Dict:
    





    seed_everything(seed)
    export_root.mkdir(parents=True, exist_ok=True)
    if class_weighting not in {"none", "balanced"}:
        raise ValueError("class_weighting must be 'none' or 'balanced'")
    if balanced_sampling not in {"none", "class-balanced"}:
        raise ValueError("balanced_sampling must be 'none' or 'class-balanced'")
    if balanced_sampling != "none" and class_weighting != "none":
        raise ValueError(
            "balanced_sampling and class_weighting are mutually exclusive (single-factor protocol)"
        )
    if checkpoint_selection not in {"best-val", "last"}:
        raise ValueError("checkpoint_selection must be 'best-val' or 'last'")
    if evaluation_scope not in {"validation-only", "formal-test"}:
        raise ValueError("evaluation_scope must be 'validation-only' or 'formal-test'")
    dataset_signature = _dataset_signature(data_root)
    run_config = {
        "dataset_signature": dataset_signature,
        "backbone": backbone,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "freeze_ratio": freeze_ratio,
        "seed": seed,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }
    if amp:
        run_config["amp"] = True
    if class_weighting != "none":
        run_config["class_weighting"] = class_weighting
    if balanced_sampling != "none":
        run_config["balanced_sampling"] = balanced_sampling
    if report_train_metrics:
        run_config["report_train_metrics"] = True
    if checkpoint_selection != "best-val":
        run_config["checkpoint_selection"] = checkpoint_selection
    if evaluation_scope != "formal-test":
        run_config["evaluation_scope"] = evaluation_scope
    if fast_training:
        run_config["performance"] = {
            "fast_training": True,
            "gpu_image_preprocessing": True,
            "channels_last": True,
            "prefetch_factor": prefetch_factor,
            "eval_num_workers": eval_num_workers,
        }
    run_config["implementation_sha256"] = {
        "training": file_sha256(Path(__file__)),
        "backbone": file_sha256(Path(__file__).parents[1] / "models" / "attention_resnet.py"),
        "reproducibility": file_sha256(Path(__file__).parents[1] / "utils" / "reproducibility.py"),
    }
    run_config["runtime"] = runtime_metadata()
    run_config["num_workers"] = num_workers
    run_signature = canonical_hash(run_config)
    metrics_path = export_root / f"{backbone}_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("run_signature") != run_signature:
            raise RuntimeError(
                f"refusing stale cache at {metrics_path}; configuration or dataset signature changed"
            )
        cached = _metrics_from_json(payload)
        checkpoint_file = payload.get("checkpoint_file", f"{backbone}_best.pt")
        checkpoint_path = export_root / checkpoint_file
        if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != payload.get("checkpoint_sha256"):
            raise RuntimeError(f"cached checkpoint is missing or changed: {checkpoint_path}")
        cached["checkpoint"] = str(export_root / checkpoint_file)
        return cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, num_classes, class_names = image_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        fast_training=fast_training,
        prefetch_factor=prefetch_factor,
        eval_num_workers=eval_num_workers,
        balanced_sampling=balanced_sampling,
        include_test=evaluation_scope == "formal-test",
    )
    model = build_backbone(backbone, num_classes=num_classes, pretrained=True)
    initial_digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        initial_digest.update(name.encode("utf-8"))
        initial_digest.update(str(tensor.dtype).encode("utf-8"))
        initial_digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        initial_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    initial_state_sha256 = initial_digest.hexdigest()
    freeze_partial_backbone(model, freeze_ratio)
    channels_last = bool(fast_training and device.type == "cuda")
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    class_weights = None
    if class_weighting == "balanced":
        class_weights = balanced_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    sampler_class_weights = None
    if balanced_sampling != "none":
        sampler_class_weights = balanced_class_weights(train_loader.dataset)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    history = {"train_loss": [], "val_accuracy": [], "val_macro_f1": [], "epoch_seconds": []}
    best_state = None
    best_acc = -1.0
    ckpt_path = export_root / f"{backbone}_checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("run_signature") != run_signature:
            raise RuntimeError(f"refusing incompatible checkpoint: {ckpt_path}")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        history = ckpt["history"]
        history.setdefault("epoch_seconds", [])
        best_acc = float(ckpt["best_acc"])
        best_state = ckpt.get("best_state")
        if "rng_state" not in ckpt:
            raise RuntimeError(f"checkpoint lacks exact-resume RNG state: {ckpt_path}")
        _restore_training_rng_state(ckpt["rng_state"], train_loader, val_loader, seed)

    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        samples_seen = 0
        batch_started = time.perf_counter()
        report_every = max(1, len(train_loader) // 10)
        for batch_index, (images, labels) in enumerate(train_loader, 1):
            images = _prepare_images(images, device, fast_training, channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            samples_seen += int(labels.shape[0])
            if batch_index % report_every == 0 or batch_index == len(train_loader):
                elapsed = max(time.perf_counter() - batch_started, 1e-9)
                speed = samples_seen / elapsed
                remaining_current = max(len(train_loader.dataset) - samples_seen, 0)
                remaining_future = max(epochs - epoch - 1, 0) * len(train_loader.dataset)
                eta = (remaining_current + remaining_future) / speed if speed else None
                print(
                    f"[train epoch {epoch + 1}/{epochs}] batch {batch_index}/{len(train_loader)}, "
                    f"speed={speed:.1f} samples/s, training ETA={_format_duration(eta)}"
                )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            progress_label=f"val epoch {epoch + 1}",
            fast_training=fast_training,
            channels_last=channels_last,
            class_names=class_names,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        history["train_loss"].append(running_loss / max(len(train_loader), 1))
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["epoch_seconds"].append(epoch_seconds)
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        _atomic_torch_save(
            {
                "run_signature": run_signature,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_acc": best_acc,
                "best_state": best_state,
                "rng_state": _capture_training_rng_state(train_loader),
            },
            ckpt_path,
        )
        average_epoch = float(np.mean(history["epoch_seconds"]))
        remaining_epochs = max(epochs - epoch - 1, 0)
        print(
            f"Epoch {epoch + 1}/{epochs}: val_acc={val_metrics['accuracy'] * 100:.2f}%, "
            f"epoch_time={_format_duration(epoch_seconds)}, "
            f"overall ETA={_format_duration(average_epoch * remaining_epochs)}"
        )

    if checkpoint_selection == "best-val":
        if best_state is None:
            raise RuntimeError("training produced no selectable checkpoint")
        model.load_state_dict(best_state)
        selected_checkpoint = "best-val"
    else:
        selected_checkpoint = "last"
    train_metrics = None
    if report_train_metrics:
        train_metrics = evaluate(
            model,
            train_loader,
            device,
            "diagnostic train",
            fast_training,
            channels_last,
            class_names,
        )
    val_metrics = evaluate(
        model,
        val_loader,
        device,
        "final validation",
        fast_training,
        channels_last,
        class_names,
    )
    test_metrics = None
    if evaluation_scope == "formal-test":
        if test_loader is None:
            raise RuntimeError("formal-test scope requires an independent-test loader")
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            "independent test",
            fast_training,
            channels_last,
            class_names,
        )
    checkpoint_file = (
        f"{backbone}_best.pt" if selected_checkpoint == "best-val" else f"{backbone}_last.pt"
    )
    _atomic_torch_save(model.state_dict(), export_root / checkpoint_file)
    plot_training_curves(history, export_root / f"{backbone}_curves.png", f"{backbone} training history")
    plot_confusion_matrix(
        val_metrics["confusion_matrix"], class_names, export_root / f"{backbone}_val_confusion_matrix.png",
        f"{backbone} validation confusion matrix",
    )
    if test_metrics is not None:
        plot_confusion_matrix(
            test_metrics["confusion_matrix"], class_names, export_root / f"{backbone}_test_confusion_matrix.png",
            f"{backbone} independent test confusion matrix",
        )
    result_payload = {
        "run_signature": run_signature,
        "run_config": run_config,
        "runtime": runtime_metadata(),
        "initial_model_state_sha256": initial_state_sha256,
        "backbone": backbone,
        "class_names": class_names,
        "history": history,
        "class_weighting": class_weighting,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "balanced_sampling": balanced_sampling,
        "checkpoint_selection": selected_checkpoint,
        "evaluation_scope": evaluation_scope,
        "independent_test_opened": test_metrics is not None,
        "dataloaded_splits": ["train", "val", *( ["test"] if test_metrics is not None else [])],
        "checkpoint_file": checkpoint_file,
        "checkpoint_sha256": file_sha256(export_root / checkpoint_file),
        "sampler_class_weights": (
            sampler_class_weights.detach().cpu().tolist() if sampler_class_weights is not None else None
        ),
        "val_metrics": val_metrics,
    }
    if test_metrics is not None:
        result_payload["test_metrics"] = test_metrics
        result_payload["paper_metrics"] = test_metrics
        result_payload["metrics"] = test_metrics
    else:
        result_payload["validation_metrics"] = val_metrics
        result_payload["formal_claim_authorized"] = False
    if train_metrics is not None:
        result_payload["train_metrics"] = train_metrics
        if balanced_sampling != "none":
            result_payload["train_metrics_note"] = "biased under balanced sampling, debug only"
    export_json(metrics_path, result_payload)
    return _metrics_from_json(result_payload) | {"checkpoint": str(export_root / checkpoint_file)}


def evaluate_classifier_checkpoint(
    data_root: Path,
    export_root: Path,
    checkpoint: Path,
    backbone: str = "resnet34_ema",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    fast_training: bool = False,
    prefetch_factor: int = 4,
    eval_num_workers: int = 0,
) -> Dict:
    
    seed_everything(seed)
    export_root.mkdir(parents=True, exist_ok=True)
    dataset_signature = _dataset_signature(data_root)
    checkpoint_hash = file_sha256(checkpoint)
    evaluation_config = {
        "dataset_signature": dataset_signature,
        "checkpoint_sha256": checkpoint_hash,
        "backbone": backbone,
        "batch_size": batch_size,
        "seed": seed,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }
    if fast_training:
        evaluation_config["performance"] = {
            "fast_training": True,
            "gpu_image_preprocessing": True,
            "channels_last": True,
            "prefetch_factor": prefetch_factor,
            "eval_num_workers": eval_num_workers,
        }
    evaluation_signature = canonical_hash(evaluation_config)
    metrics_path = export_root / f"{backbone}_evaluation_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("evaluation_signature") != evaluation_signature:
            raise RuntimeError(f"refusing stale evaluation cache: {metrics_path}")
        return _metrics_from_json(payload)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader, num_classes, class_names = image_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        fast_training=fast_training,
        prefetch_factor=prefetch_factor,
        eval_num_workers=eval_num_workers,
    )
    model = build_backbone(backbone, num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    channels_last = bool(fast_training and device.type == "cuda")
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    val_metrics = evaluate(
        model,
        val_loader,
        device,
        "fixed-model validation",
        fast_training,
        channels_last,
        class_names,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        "fixed-model noisy test",
        fast_training,
        channels_last,
        class_names,
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        export_root / f"{backbone}_test_confusion_matrix.png",
        f"{backbone} fixed-model independent test confusion matrix",
    )
    payload = {
        "run_signature": evaluation_signature,
        "evaluation_signature": evaluation_signature,
        "evaluation_config": evaluation_config,
        "runtime": runtime_metadata(),
        "backbone": backbone,
        "class_names": class_names,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "paper_metrics": test_metrics,
        "metrics": test_metrics,
    }
    export_json(metrics_path, payload)
    return _metrics_from_json(payload)
