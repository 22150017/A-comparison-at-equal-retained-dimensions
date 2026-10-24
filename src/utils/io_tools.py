from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_json(path: Path, payload: dict) -> None:
    
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def stratified_split_indices(labels: Sequence[str], train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=np.array(labels),
    )
    return train_idx, val_idx


def stratified_split_three_way(
    labels: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    indices = np.arange(len(labels))
    test_ratio = 1.0 - train_ratio - val_ratio
    
    train_idx, rest_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=np.array(labels),
    )
    
    rest_labels = [labels[i] for i in rest_idx]
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx_rel, test_idx_rel = train_test_split(
        np.arange(len(rest_idx)),
        train_size=val_size,
        random_state=seed,
        stratify=np.array(rest_labels),
    )
    val_idx = rest_idx[val_idx_rel]
    test_idx = rest_idx[test_idx_rel]
    return train_idx, val_idx, test_idx


def stratified_group_split_three_way(
    labels: Sequence[str],
    groups: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if len(labels) != len(groups) or not labels:
        raise ValueError("labels and groups must be non-empty and have equal length")
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, val_ratio and test_ratio must all be positive")

    group_label: dict[str, str] = {}
    class_groups: dict[str, list[str]] = {}
    for label, group in zip(labels, groups):
        group = str(group)
        previous = group_label.setdefault(group, str(label))
        if previous != label:
            raise ValueError(f"group {group!r} contains multiple labels: {previous!r}, {label!r}")
    for group, label in group_label.items():
        class_groups.setdefault(label, []).append(group)

    rng = np.random.default_rng(seed)
    split_groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for label, units in sorted(class_groups.items()):
        units = list(units)
        rng.shuffle(units)
        if len(units) < 3:
            raise ValueError(
                f"class {label!r} has only {len(units)} independent groups; "
                "at least 3 are required for train/validation/test"
            )
        n_val = max(1, int(round(len(units) * val_ratio)))
        n_test = max(1, int(round(len(units) * test_ratio)))
        if n_val + n_test >= len(units):
            overflow = n_val + n_test - (len(units) - 1)
            if n_val >= n_test:
                n_val -= overflow
            else:
                n_test -= overflow
        split_groups["val"].update(units[:n_val])
        split_groups["test"].update(units[n_val : n_val + n_test])
        split_groups["train"].update(units[n_val + n_test :])

    group_array = np.asarray([str(group) for group in groups])
    arrays = [
        np.flatnonzero(np.isin(group_array, sorted(split_groups[name])))
        for name in ("train", "val", "test")
    ]
    assert_disjoint_groups(groups, arrays[0], arrays[1], arrays[2])
    return arrays[0], arrays[1], arrays[2]


def assert_disjoint_groups(
    groups: Sequence[str],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
) -> None:
    
    group_array = np.asarray([str(group) for group in groups])
    split_sets = [set(group_array[np.asarray(idx, dtype=int)]) for idx in (train_idx, val_idx, test_idx)]
    names = ("train", "val", "test")
    for left in range(3):
        for right in range(left + 1, 3):
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(
                    f"group leakage between {names[left]} and {names[right]}: {sorted(overlap)[:5]}"
                )


def explicit_holdout_split_three_way(
    labels: Sequence[str],
    groups: Sequence[str],
    test_groups: Sequence[str],
    val_groups: Sequence[str],
) -> tuple[dict[int, str], dict]:
    





    if len(labels) != len(groups) or not labels:
        raise ValueError("labels and groups must be non-empty and have equal length")
    test_set = {str(group) for group in test_groups}
    val_set = {str(group) for group in val_groups}
    if not test_set or not val_set:
        raise ValueError("explicit holdout requires non-empty test_groups and val_groups")
    if test_set & val_set:
        raise ValueError(f"test and val holdout groups overlap: {sorted(test_set & val_set)}")
    available = {str(group) for group in groups}
    missing = (test_set | val_set) - available
    if missing:
        raise ValueError(f"holdout groups not present in dataset: {sorted(missing)}")
    group_label: dict[str, str] = {}
    for label, group in zip(labels, groups):
        group = str(group)
        previous = group_label.setdefault(group, str(label))
        if previous != str(label):
            raise ValueError(f"group {group!r} contains multiple labels: {previous!r} vs {label!r}")
    all_labels = set(group_label.values())
    test_labels = {group_label[group] for group in test_set}
    if test_labels != all_labels:
        raise ValueError(
            f"test holdout must cover every class; missing: {sorted(all_labels - test_labels)}"
        )
    train_groups = available - test_set - val_set
    train_labels = {group_label[group] for group in train_groups}
    if train_labels != all_labels:
        raise ValueError(
            f"training split must cover every class; missing: {sorted(all_labels - train_labels)}"
        )
    split_map: dict[int, str] = {}
    for index, group in enumerate(groups):
        name = str(group)
        split_map[index] = "test" if name in test_set else "val" if name in val_set else "train"
    record = {
        "mode": "explicit_bearing_holdout",
        "test_groups": sorted(test_set),
        "val_groups": sorted(val_set),
        "val_classes": sorted({group_label[group] for group in val_set}),
        "train_classes": sorted(train_labels),
    }
    return split_map, record


def plot_training_curves(history: dict, output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.get("train_loss", []), label="Train loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.get("val_accuracy", []), label="Validation accuracy", color="tab:orange")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], output_path: Path, title: str) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
