




from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cwru_loader import load_cwru_segments
from src.data.pu_loader import load_pu_segments
from src.features.cwt import signal_to_cwt_image
from src.reconstruction.diagnostic_prior import (
    fit_or_load_diagnostic_prior,
    select_training_prior_indices,
)
from src.reconstruction.compression_baselines import (
    downsample_interpolate,
    fit_pca_baseline,
    load_pca_baseline,
    random_projection_reconstruct,
    save_pca_baseline,
)
from src.reconstruction.sparse_recovery import (
    get_reconstructors,
    reconstruction_error,
    set_cs_seed,
    set_diagnostic_prior,
    set_gradient_lambda,
    set_sparse_basis,
    snr_db,
)
from src.training.train_classifier import train_classifier
from src.utils.io_tools import (
    explicit_holdout_split_three_way,
    export_json,
    stratified_group_split_three_way,
    stratified_split_three_way,
)
from src.utils.reproducibility import canonical_hash, file_sha256, runtime_metadata, seed_everything


_SAMPLE_CACHE: dict[str, list] = {}
_WORKER_SETTINGS: dict = {}
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts_v2"
_SOURCE_HASH_CACHE_PATH = DEFAULT_ARTIFACT_ROOT / "provenance" / "source_hash_cache.json"


def _normalized_artifact_roots(
    artifact_root: Path | str | None,
    reuse_artifact_roots: tuple[Path | str, ...] = (),
) -> tuple[Path, tuple[Path, ...]]:
    primary = Path(artifact_root or (ROOT / "artifacts_v2")).resolve()
    if primary == Path(primary.anchor):
        raise ValueError("artifact root must not be a drive root")
    ordered: list[Path] = []
    for value in (primary, *reuse_artifact_roots):
        resolved = Path(value).resolve()
        if resolved == Path(resolved.anchor):
            raise ValueError("artifact reuse root must not be a drive root")
        if resolved not in ordered:
            ordered.append(resolved)
    return primary, tuple(ordered)


def _artifact_root_for_dataset(data_root: Path, candidates: tuple[Path, ...]) -> Path:
    resolved = data_root.resolve()
    for candidate in candidates:
        try:
            resolved.relative_to((candidate / "datasets").resolve())
            return candidate
        except ValueError:
            continue
    raise ValueError(f"dataset root is outside configured artifact roots: {data_root}")


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


def _sample_identity(sample) -> str:
    return canonical_hash(
        {
            "source": str(Path(sample.source_file).resolve()),
            "segment_id": sample.segment_id,
            "group_id": sample.group_id,
        },
        length=24,
    )


def _operating_condition(sample) -> str:
    parts = Path(sample.source_file).stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else "unknown"


def _cross_condition_subset(
    samples: list,
    train_conditions: tuple[str, ...],
    test_condition: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list, dict[int, str]]:
    
    if not train_conditions or test_condition in train_conditions:
        raise ValueError("cross-condition train conditions must be non-empty and exclude test condition")
    available: dict[str, set[str]] = defaultdict(set)
    group_label: dict[str, str] = {}
    for sample in samples:
        condition = _operating_condition(sample)
        available[condition].add(str(sample.group_id))
        previous = group_label.setdefault(str(sample.group_id), sample.label)
        if previous != sample.label:
            raise ValueError(f"group {sample.group_id!r} contains multiple labels")
    missing = (set(train_conditions) | {test_condition}) - set(available)
    if missing:
        raise ValueError(f"cross-condition data are missing: {sorted(missing)}")

    eligible_groups = set.intersection(
        *(available[condition] for condition in (*train_conditions, test_condition))
    )
    by_label: dict[str, list[str]] = defaultdict(list)
    for group in eligible_groups:
        by_label[group_label[group]].append(group)
    rng = np.random.default_rng(seed)
    split_groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    test_ratio = 1.0 - train_ratio - val_ratio
    for label, groups in sorted(by_label.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(
                f"cross-condition class {label!r} has only {len(groups)} eligible bearings; "
                "at least 3 are required"
            )
        n_val = max(1, int(round(len(groups) * val_ratio)))
        n_test = max(1, int(round(len(groups) * test_ratio)))
        if n_val + n_test >= len(groups):
            overflow = n_val + n_test - (len(groups) - 1)
            if n_val >= n_test:
                n_val -= overflow
            else:
                n_test -= overflow
        split_groups["val"].update(groups[:n_val])
        split_groups["test"].update(groups[n_val : n_val + n_test])
        split_groups["train"].update(groups[n_val + n_test :])

    selected: list = []
    split_map: dict[int, str] = {}
    for sample in samples:
        group = str(sample.group_id)
        condition = _operating_condition(sample)
        split = None
        if group in split_groups["train"] and condition in train_conditions:
            split = "train"
        elif group in split_groups["val"] and condition in train_conditions:
            split = "val"
        elif group in split_groups["test"] and condition == test_condition:
            split = "test"
        if split is not None:
            split_map[len(selected)] = split
            selected.append(sample)
    for split in ("train", "val", "test"):
        labels = {sample.label for index, sample in enumerate(selected) if split_map[index] == split}
        if labels != set(by_label):
            raise ValueError(f"cross-condition {split} split lacks classes: {set(by_label) - labels}")
    return selected, split_map


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _pu_label_schema(signal_cfg: dict) -> str:
    declared = signal_cfg.get("pu_label_schema")
    if declared is not None:
        return str(declared)
    classes = set(signal_cfg["classes"])
    return "legacy_v1" if "ball" in classes and "compound" not in classes else "official_v2"


def load_samples(cfg: dict, dataset: str):
    if (dataset == "cwru" and "cwru_source_inventory" not in cfg["signal"]
            and cfg["signal"].get("cwru_loader_version") != "legacy_v1"):
        raise ValueError("CWRU requires a corrected source inventory. Use run_cwru_corrected_benchmark_v2.py; legacy_v1 is only for historical reproduction.")
    cache_key = canonical_hash({"dataset": dataset, "paths": cfg["paths"], "signal": cfg["signal"]})
    if cache_key in _SAMPLE_CACHE:
        print(f"reusing in-memory {dataset} samples")
        return _SAMPLE_CACHE[cache_key]
    signal_cfg = cfg["signal"]
    if dataset == "pu":
        samples = load_pu_segments(
            Path(cfg["paths"]["pu_root"]),
            segment_length=signal_cfg["segment_length"],
            stride=signal_cfg["stride"],
            normalize=signal_cfg["normalize"],
            allowed_classes=signal_cfg["classes"],
            target_sampling_rate=signal_cfg["sampling_rate"],
            label_schema=_pu_label_schema(signal_cfg),
        )
    elif dataset == "cwru":
        if "cwru_source_inventory" in signal_cfg:
            from src.data.cwru_corrected import load_corrected_segments

            samples = load_corrected_segments(
                Path(cfg["paths"]["cwru_root"]), signal_cfg["cwru_source_inventory"],
                segment_length=signal_cfg["segment_length"], stride=signal_cfg["stride"],
                normalize=signal_cfg["normalize"],
            )
        else:
            samples = load_cwru_segments(
                Path(cfg["paths"]["cwru_root"]),
                segment_length=signal_cfg["segment_length"],
                stride=signal_cfg["stride"],
                normalize=signal_cfg["normalize"],
                allowed_classes=signal_cfg["classes"],
                allowed_sampling_rate=signal_cfg["sampling_rate"],
                allowed_path_keywords=signal_cfg.get("allowed_path_keywords"),
            )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    _SAMPLE_CACHE[cache_key] = samples
    return samples


def _source_fingerprints(samples) -> list[dict]:
    cache: dict[str, dict] = {}
    if _SOURCE_HASH_CACHE_PATH.exists():
        try:
            cache = json.loads(_SOURCE_HASH_CACHE_PATH.read_text(encoding="utf-8")).get("files", {})
        except (json.JSONDecodeError, OSError):
            cache = {}
    records = []
    changed = False
    for source in sorted({str(Path(sample.source_file).resolve()) for sample in samples}):
        path = Path(source)
        stat = path.stat()
        cached = cache.get(source)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            record = cached
        else:
            record = {
                "path": source,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path),
            }
            cache[source] = record
            changed = True
        records.append(record)
    if changed:
        export_json(_SOURCE_HASH_CACHE_PATH, {"files": cache})
    return records


def _valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _link_or_copy(source: Path, destination: Path) -> None:
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _initialize_prepare_worker(settings: dict) -> None:
    global _WORKER_SETTINGS
    defaults = {
        "compression_baseline": None,
        "no_cs": False,
        "pca_model": None,
        "method": "ilsomp",
        "ratio": 0.5,
        "sparsity_k": 64,
        "gradient_lambda": 0.25,
        "diagnostic_prior_mu": 0.5,
        "diagnostic_prior_weights": None,
        "sparse_basis": "dct",
        "representation": "cwt",
        "wavelet": "morl",
        "cwt_scales_start": 1,
        "cwt_scales_end": 128,
        "stft_nperseg": 64,
        "stft_noverlap": 32,
        "noise_snr_db": None,
    }
    for key, value in defaults.items():
        settings.setdefault(key, value)
    _WORKER_SETTINGS = settings
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass
    if settings["compression_baseline"] is None and not settings["no_cs"]:
        set_cs_seed(settings["seed"])
        set_sparse_basis(settings["sparse_basis"])
        set_gradient_lambda(settings["gradient_lambda"])
        set_diagnostic_prior(
            settings.get("diagnostic_prior_weights"),
            settings.get("diagnostic_prior_mu", 0.5),
        )


def _process_image_task(task: tuple[dict, np.ndarray]) -> dict:
    record, signal = task
    output_path = Path(record["image_path"])
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    cache_temporary: Path | None = None
    try:
        cache_path = Path(record["cache_path"]) if record.get("cache_path") else None
        if cache_path is not None and _valid_png(cache_path):
            metadata_path = cache_path.with_suffix(".json")
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    for optional in ("reconstruction_error", "snr_db"):
                        if optional in metadata:
                            record[optional] = metadata[optional]
                except (json.JSONDecodeError, OSError, TypeError, ValueError):
                    pass
            _link_or_copy(cache_path, output_path)
            record["status"] = "reused_valid_image"
            return record
        processed = np.asarray(signal, dtype=np.float64)
        noise_snr_db = _WORKER_SETTINGS.get("noise_snr_db")
        if noise_snr_db is not None and record["split"] == "test":
            signal_power = float(np.mean(processed ** 2))
            noise_power = signal_power / (10.0 ** (float(noise_snr_db) / 10.0))
            noise_seed = (_WORKER_SETTINGS["seed"] ^ int(record["sample_id"][:8], 16)) % (2 ** 32)
            noise_rng = np.random.default_rng(noise_seed)
            processed = processed + noise_rng.normal(0.0, np.sqrt(noise_power), size=len(processed))
            record["noise_seed"] = int(noise_seed)

        baseline = _WORKER_SETTINGS["compression_baseline"]
        if baseline == "downsample":
            reconstructed = downsample_interpolate(processed, _WORKER_SETTINGS["ratio"])
        elif baseline == "random_projection":
            reconstructed = random_projection_reconstruct(
                processed, _WORKER_SETTINGS["ratio"], _WORKER_SETTINGS["seed"]
            )
        elif baseline == "pca":
            reconstructed = _WORKER_SETTINGS["pca_model"].reconstruct(processed)
        else:
            reconstructed = None
        if reconstructed is not None:
            record.update(
                {
                    "reconstruction_error": reconstruction_error(processed, reconstructed),
                    "snr_db": snr_db(processed, reconstructed),
                }
            )
            processed = reconstructed
        elif not _WORKER_SETTINGS["no_cs"]:
            reconstructor = get_reconstructors()[_WORKER_SETTINGS["method"]]
            result = reconstructor(
                processed,
                _WORKER_SETTINGS["ratio"],
                _WORKER_SETTINGS["sparsity_k"],
            )
            processed = result.reconstructed_signal
            record.update({"reconstruction_error": result.reconstruction_error, "snr_db": result.snr_db})
        image = (
            signal_to_cwt_image(
                processed,
                _WORKER_SETTINGS["wavelet"],
                scales=np.arange(
                    _WORKER_SETTINGS["cwt_scales_start"],
                    _WORKER_SETTINGS["cwt_scales_end"] + 1,
                ),
                image_size=_WORKER_SETTINGS["image_size"],
            )
            if _WORKER_SETTINGS["representation"] == "cwt"
            else _stft_image(
                processed,
                _WORKER_SETTINGS["image_size"],
                _WORKER_SETTINGS["stft_nperseg"],
                _WORKER_SETTINGS["stft_noverlap"],
            )
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            image.save(cache_temporary, format="PNG")
            cache_temporary.replace(cache_path)
            cache_metadata = {
                key: record[key]
                for key in ("reconstruction_error", "snr_db")
                if key in record
            }
            if cache_metadata:
                export_json(cache_path.with_suffix(".json"), cache_metadata)
            _link_or_copy(cache_path, output_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(temporary, format="PNG")
            temporary.replace(output_path)
        record["status"] = "generated"
        record.pop("error", None)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        if cache_temporary is not None and cache_temporary.exists():
            cache_temporary.unlink()
        record.update({"status": "failed", "error": repr(exc)})
    return record


def _load_progress(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"ignoring truncated progress line {line_number}: {path}")
                continue
            if record.get("sample_id"):
                records[record["sample_id"]] = record
    return records


def _stft_image(signal: np.ndarray, image_size: int, nperseg: int = 64, noverlap: int = 32) -> Image.Image:
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("STFT requires a nonempty finite one-dimensional signal")
    if image_size <= 0 or nperseg <= 0 or noverlap < 0:
        raise ValueError("invalid STFT or image dimensions")
    segment = min(nperseg, len(signal))
    overlap = min(noverlap, segment - 1)
    _, _, spectrum = stft(signal, fs=12000, nperseg=segment, noverlap=overlap)
    magnitude = np.log1p(np.abs(spectrum))
    magnitude = magnitude / (magnitude.max() + 1e-12)
    from matplotlib import colormaps

    rgb = (colormaps["viridis"](magnitude[::-1])[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((image_size, image_size), Image.Resampling.BILINEAR)


def prepare_dataset(
    cfg: dict,
    dataset: str,
    method: str,
    ratio: float,
    representation: str,
    seed: int,
    no_cs: bool = False,
    max_samples_per_group: int | None = None,
    prepare_workers: int = 1,
    prepare_chunksize: int = 8,
    sparsity_k: int | None = None,
    gradient_lambda: float = 0.25,
    diagnostic_prior_mu: float = 0.5,
    diagnostic_prior_max_per_class: int = 4096,
    sparse_basis: str = "dct",
    wavelet: str | None = None,
    cwt_scales_start: int = 1,
    cwt_scales_end: int = 128,
    stft_nperseg: int = 64,
    stft_noverlap: int | None = None,
    noise_snr_db: float | None = None,
    train_conditions: tuple[str, ...] | None = None,
    test_condition: str | None = None,
    compression_baseline: str | None = None,
    pca_fit_samples: int = 4096,
    test_groups: tuple[str, ...] | None = None,
    val_groups: tuple[str, ...] | None = None,
    split_unit: str = "bearing",
    artifact_root: Path | str | None = None,
    reuse_artifact_roots: tuple[Path | str, ...] = (),
) -> Path:
    seed_everything(seed)
    sparsity_k = int(sparsity_k if sparsity_k is not None else cfg["compressed_sensing"]["sparsity_k"])
    wavelet = str(wavelet or cfg["cwt"]["wavelet"])
    stft_noverlap = int(stft_noverlap if stft_noverlap is not None else stft_nperseg // 2)
    train_conditions = tuple(train_conditions or ())
    if split_unit not in {"bearing", "source-file", "window"}:
        raise ValueError("split_unit must be 'bearing', 'source-file', or 'window'")
    allowed_baselines = {None, "downsample", "pca", "random_projection"}
    if compression_baseline not in allowed_baselines:
        raise ValueError(f"unsupported compression baseline: {compression_baseline}")
    if no_cs and compression_baseline is not None:
        raise ValueError("--no-cs and --compression-baseline are mutually exclusive")
    if sparsity_k < 1 or gradient_lambda < 0:
        raise ValueError("sparsity_k must be positive and gradient_lambda non-negative")
    if diagnostic_prior_mu < 0 or diagnostic_prior_max_per_class < 1:
        raise ValueError("diagnostic prior mu must be non-negative and its sample cap positive")
    if method == "dp_ilsomp" and (no_cs or compression_baseline is not None):
        raise ValueError("dp_ilsomp cannot be combined with --no-cs or a compression baseline")
    if method == "dp_ilsomp" and sparse_basis.lower() != "dct":
        raise ValueError("dp_ilsomp currently requires the DCT sparse basis")
    if cwt_scales_start < 1 or cwt_scales_end < cwt_scales_start:
        raise ValueError("invalid CWT scale range")
    if stft_nperseg < 2 or not 0 <= stft_noverlap < stft_nperseg:
        raise ValueError("STFT requires nperseg >= 2 and 0 <= noverlap < nperseg")
    if bool(train_conditions) != bool(test_condition):
        raise ValueError("cross-condition mode requires both train_conditions and test_condition")
    if train_conditions and dataset != "pu":
        raise ValueError("cross-condition mode currently requires the Paderborn dataset")
    if (test_groups or val_groups) and train_conditions:
        raise ValueError("explicit holdout is incompatible with cross-condition mode")
    if split_unit == "source-file" and train_conditions:
        raise ValueError("--split-unit source-file is incompatible with cross-condition mode")
    if split_unit == "source-file" and (test_groups or val_groups):
        raise ValueError("--split-unit source-file is incompatible with explicit bearing holdout")
    if (test_groups or val_groups) and max_samples_per_group:
        raise ValueError("explicit holdout requires full per-bearing counts; drop --max-samples-per-group")

    samples = load_samples(cfg, dataset)
    if not samples:
        raise RuntimeError(f"no samples loaded for {dataset}")
    if any(not sample.group_id for sample in samples):
        raise RuntimeError("every sample must have a group_id")
    if train_conditions:
        samples, split_map = _cross_condition_subset(
            samples,
            train_conditions,
            str(test_condition),
            cfg["training"]["train_ratio"],
            cfg["training"]["val_ratio"],
            seed,
        )
    else:
        split_map = {}
    if max_samples_per_group:
        kept = []
        kept_splits: dict[int, str] = {}
        counts: Counter[str] = Counter()
        for index, sample in enumerate(samples):
            if counts[str(sample.group_id)] < max_samples_per_group:
                if split_map:
                    kept_splits[len(kept)] = split_map[index]
                kept.append(sample)
                counts[str(sample.group_id)] += 1
        samples = kept
        if split_map:
            split_map = kept_splits

    holdout_record = None
    split_group_ids = [
        str(Path(sample.source_file).resolve())
        if split_unit in {"source-file", "window"}
        else str(sample.group_id)
        for sample in samples
    ]
    if not split_map:
        labels = [sample.label for sample in samples]
        groups = split_group_ids
        if test_groups or val_groups:
            split_map, holdout_record = explicit_holdout_split_three_way(
                labels, groups, list(test_groups or ()), list(val_groups or ())
            )
        elif split_unit == "window":
            train_idx, val_idx, test_idx = stratified_split_three_way(
                labels,
                cfg["training"]["train_ratio"],
                cfg["training"]["val_ratio"],
                seed,
            )
            split_map = {
                **{int(index): "train" for index in train_idx},
                **{int(index): "val" for index in val_idx},
                **{int(index): "test" for index in test_idx},
            }
        else:
            train_idx, val_idx, test_idx = stratified_group_split_three_way(
                labels,
                groups,
                cfg["training"]["train_ratio"],
                cfg["training"]["val_ratio"],
                seed,
            )
            split_map = {
                **{int(index): "train" for index in train_idx},
                **{int(index): "val" for index in val_idx},
                **{int(index): "test" for index in test_idx},
            }
    sample_ids = [_sample_identity(sample) for sample in samples]
    source_files = _source_fingerprints(samples)
    phase2: dict = {}
    diagnostic_prior_indices: list[int] = []
    if method == "dp_ilsomp" and not no_cs and compression_baseline is None:
        diagnostic_prior_indices = select_training_prior_indices(
            [str(sample.label) for sample in samples],
            split_map,
            sample_ids,
            seed,
            diagnostic_prior_max_per_class,
        )
        phase2["diagnostic_prior"] = {
            "version": 1,
            "fit_scope": "training_only",
            "formula": "class-balanced Fisher on normalized DCT magnitudes",
            "mu": float(diagnostic_prior_mu),
            "max_samples_per_class": int(diagnostic_prior_max_per_class),
            "fit_sample_count": len(diagnostic_prior_indices),
            "fit_sample_ids_hash": canonical_hash(
                [sample_ids[index] for index in diagnostic_prior_indices], length=32
            ),
        }
    if sparse_basis.lower() != "dct":
        phase2["sparse_basis"] = sparse_basis.lower()
    if not np.isclose(gradient_lambda, 0.25):
        phase2["gradient_lambda"] = float(gradient_lambda)
    if wavelet.lower() not in {"morl", "morlet"}:
        phase2["cwt_wavelet"] = wavelet
    if (cwt_scales_start, cwt_scales_end) != (1, 128):
        phase2["cwt_scales"] = [cwt_scales_start, cwt_scales_end]
    if (stft_nperseg, stft_noverlap) != (64, 32):
        phase2["stft"] = {"nperseg": stft_nperseg, "noverlap": stft_noverlap}
    if noise_snr_db is not None:
        phase2["noise"] = {"snr_db": float(noise_snr_db), "scope": "test_only"}
    if train_conditions:
        phase2["cross_condition"] = {
            "train_conditions": list(train_conditions),
            "test_condition": test_condition,
            "validation_domain": "source",
            "bearing_groups_disjoint": True,
        }
    if compression_baseline is not None:
        train_ids = [sample_ids[index] for index in range(len(samples)) if split_map[index] == "train"]
        rng = np.random.default_rng(seed)
        selected = np.arange(len(train_ids))
        if len(selected) > pca_fit_samples:
            selected = np.sort(rng.choice(selected, size=pca_fit_samples, replace=False))
        phase2["compression_baseline"] = compression_baseline
        if compression_baseline == "pca":
            phase2["pca"] = {
                "fit_scope": "training_only",
                "max_fit_samples": pca_fit_samples,
                "fit_sample_count": len(selected),
                "fit_sample_ids_hash": canonical_hash([train_ids[index] for index in selected]),
            }
    split_condition = {
        "mode": "explicit_bearing_holdout",
        "unit": "bearing_id" if dataset == "pu" else "source_recording",
    } if holdout_record else {
        "mode": (
            "deliberately_leaky_window_control"
            if split_unit == "window"
            else (
                "stratified_source_file_random"
                if split_unit == "source-file"
                else "stratified_group_random"
            )
        ),
        "train_ratio": cfg["training"]["train_ratio"],
        "val_ratio": cfg["training"]["val_ratio"],
        "test_ratio": 1 - cfg["training"]["train_ratio"] - cfg["training"]["val_ratio"],
        "unit": (
            "window_invalid_control"
            if split_unit == "window"
            else ("source_file"
            if split_unit == "source-file"
            else ("bearing_id" if dataset == "pu" else "source_recording")
            )
        ),
        "paper_claim_authorized": split_unit != "window",
        "warning": (
            "Deliberately leaky window-level control; never report as valid generalization performance."
            if split_unit == "window"
            else None
        ),
    }
    condition = {
        "schema_version": 2,
        "dataset": dataset,
        "method": None if no_cs else (f"baseline:{compression_baseline}" if compression_baseline else method),
        "compression_ratio": None if no_cs else ratio,
        "sparsity_k": None if no_cs or compression_baseline else sparsity_k,
        "representation": representation,
        "seed": seed,
        "signal": cfg["signal"],
        "split": split_condition,
        "sources": source_files,
        "max_samples_per_group": max_samples_per_group,
    }
    if phase2:
        condition["phase2"] = phase2
    if holdout_record:
        condition["explicit_holdout"] = holdout_record
    signature = canonical_hash(condition)
    method_name = "no_cs" if no_cs else (f"baseline_{compression_baseline}" if compression_baseline else method)
    condition_name = f"{method_name}_{representation}_r{ratio:.2f}_s{seed}"
    primary_artifact_root, artifact_roots = _normalized_artifact_roots(
        artifact_root, reuse_artifact_roots
    )
    dataset_relative = Path("datasets") / dataset / condition_name / signature
    output_root = primary_artifact_root / dataset_relative
    selected_artifact_root = primary_artifact_root
    for candidate_root in artifact_roots:
        candidate = candidate_root / dataset_relative
        if candidate.exists():
            output_root = candidate
            selected_artifact_root = candidate_root
            break
    manifest_path = output_root / "dataset_manifest.json"
    state_path = output_root / "dataset_state.json"
    progress_path = output_root / "dataset_progress.jsonl"
    resume_path = output_root / "resume_status.json"
    diagnostic_prior_weights = None
    diagnostic_prior_metadata = None
    if method == "dp_ilsomp" and not no_cs and compression_baseline is None:
        diagnostic_prior_weights, diagnostic_prior_metadata = fit_or_load_diagnostic_prior(
            output_root,
            samples,
            split_map,
            sample_ids,
            seed,
            diagnostic_prior_max_per_class,
        )
    previous_manifest: dict = {}
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("signature") != signature:
            raise RuntimeError(f"incompatible existing dataset: {output_root}")
        prune_receipt = output_root / "rejected_diagnostic_prune_receipt.json"
        if prune_receipt.exists():
            raise RuntimeError(
                f"dataset was intentionally archived as a rejected diagnostic: {prune_receipt}"
            )
        if previous_manifest.get("complete"):
            print(f"dataset already complete: {output_root}")
            return output_root
        print(f"resuming incomplete dataset: {output_root}")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise RuntimeError(f"resume state signature mismatch: {state_path}")
    else:
        export_json(state_path, {"signature": signature, "condition": condition, "complete": False})

    if prepare_workers < 1 or prepare_chunksize < 1:
        raise ValueError("prepare_workers and prepare_chunksize must be positive")
    image_size = int(cfg["cwt"]["image_size"])
    pca_model = None
    pca_model_sha256 = None
    if compression_baseline == "pca":
        pca_path = output_root / "pca_training_only.npz"
        if pca_path.exists():
            pca_model = load_pca_baseline(pca_path)
        else:
            train_positions = [index for index in range(len(samples)) if split_map[index] == "train"]
            pca_model = fit_pca_baseline(
                [samples[index].signal for index in train_positions],
                [sample_ids[index] for index in train_positions],
                ratio,
                seed,
                pca_fit_samples,
            )
            save_pca_baseline(pca_path, pca_model)
        pca_model_sha256 = file_sha256(pca_path)
        export_json(
            output_root / "pca_training_only.json",
            {
                "fit_scope": "training_only",
                "fit_sample_ids": list(pca_model.fit_sample_ids),
                "requested_components": pca_model.requested_components,
                "actual_components": pca_model.actual_components,
                "sha256": pca_model_sha256,
            },
        )
    settings = {
        "no_cs": no_cs,
        "method": method,
        "ratio": ratio,
        "sparsity_k": sparsity_k,
        "gradient_lambda": float(gradient_lambda),
        "diagnostic_prior_mu": float(diagnostic_prior_mu),
        "diagnostic_prior_weights": diagnostic_prior_weights,
        "sparse_basis": sparse_basis,
        "compression_baseline": compression_baseline,
        "pca_model": pca_model,
        "representation": representation,
        "wavelet": wavelet,
        "cwt_scales_start": cwt_scales_start,
        "cwt_scales_end": cwt_scales_end,
        "stft_nperseg": stft_nperseg,
        "stft_noverlap": stft_noverlap,
        "noise_snr_db": noise_snr_db,
        "image_size": image_size,
        "seed": seed,
    }
    source_sha = {record["path"]: record["sha256"] for record in source_files}
    cache_transform = {
        "version": 2,
        "method": None if no_cs else method,
        "no_cs": no_cs,
        "compression_baseline": compression_baseline,
        "ratio": None if no_cs else ratio,
        "sparsity_k": None if no_cs or compression_baseline else sparsity_k,
        "gradient_lambda": gradient_lambda if method in {"ilsomp", "dp_ilsomp"} and not no_cs else None,
        "diagnostic_prior_mu": diagnostic_prior_mu if method == "dp_ilsomp" and not no_cs else None,
        "diagnostic_prior_sha256": (
            diagnostic_prior_metadata.get("npz_sha256") if diagnostic_prior_metadata else None
        ),
        "sparse_basis": sparse_basis if not no_cs and not compression_baseline else None,
        "representation": representation,
        "wavelet": wavelet if representation == "cwt" else None,
        "cwt_scales": [cwt_scales_start, cwt_scales_end] if representation == "cwt" else None,
        "stft": [stft_nperseg, stft_noverlap] if representation == "stft" else None,
        "image_size": image_size,
        "signal": cfg["signal"],
        "seed": seed if not no_cs or compression_baseline == "random_projection" else None,
        "pca_model_sha256": pca_model_sha256,
    }
    progress_records = _load_progress(progress_path)
    for record in previous_manifest.get("records", []):
        if record.get("sample_id"):
            progress_records.setdefault(record["sample_id"], record)
    record_map: dict[str, dict] = {}
    ordered_ids: list[str] = []
    pending: list[tuple[dict, np.ndarray]] = []
    recovered_without_log: list[dict] = []
    for index, sample in enumerate(samples):
        split = split_map[index]
        identity = sample_ids[index]
        output_path = output_root / split / sample.label / f"{identity}.png"
        resolved_source = str(Path(sample.source_file).resolve())
        record = {
            "sample_id": identity,
            "source_file": resolved_source,
            "segment_id": sample.segment_id,
            "group_id": str(sample.group_id),
            "split_group_id": split_group_ids[index],
            "label": sample.label,
            "split": split,
            "image_path": str(output_path),
        }
        if dataset == "cwru" and "cwru_source_inventory" in cfg["signal"]:
            record["signal_key"] = sample.sensor
            record["sampling_rate_hz"] = sample.sampling_rate
        output_is_valid = _valid_png(output_path)
        sample_transform = dict(cache_transform)
        sample_transform["noise_snr_db"] = noise_snr_db if split == "test" else None
        source_digest = source_sha.get(resolved_source) or source_sha.get(str(sample.source_file))
        if source_digest is None:
            raise RuntimeError(f"missing source fingerprint for {resolved_source}")
        cache_key = canonical_hash(
            {
                "sample_id": identity,
                "source_sha256": source_digest,
                "transform": sample_transform,
            },
            length=32,
        )
        cache_path = (
            selected_artifact_root / "image_cache" / "v2"
            / cache_key[:2] / f"{cache_key}.png"
        )
        record["cache_path"] = str(cache_path)
        cache_metadata_path = cache_path.with_suffix(".json")
        cache_is_valid = _valid_png(cache_path)
        if output_is_valid and not cache_is_valid:
            _link_or_copy(output_path, cache_path)
        elif not output_is_valid and cache_is_valid:
            _link_or_copy(cache_path, output_path)
            output_is_valid = True
        ordered_ids.append(identity)
        existing = progress_records.get(identity)
        if output_is_valid:
            if existing:
                for optional in ("reconstruction_error", "snr_db"):
                    if optional in existing:
                        record[optional] = existing[optional]
                record["status"] = (
                    existing.get("status")
                    if existing.get("status") in {"generated", "reused_valid_image"}
                    else "reused_valid_image"
                )
            else:
                record["status"] = "reused_valid_image"
                recovered_without_log.append(record)
            if not any(optional in record for optional in ("reconstruction_error", "snr_db")) and cache_metadata_path.is_file():
                try:
                    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
                    for optional in ("reconstruction_error", "snr_db"):
                        if optional in cache_metadata:
                            record[optional] = cache_metadata[optional]
                except (json.JSONDecodeError, OSError, TypeError, ValueError):
                    pass
            record_map[identity] = record
        else:
            pending.append((record, sample.signal))

    output_root.mkdir(parents=True, exist_ok=True)
    processed_count = len(record_map)
    generated_count = 0
    failed_count = 0
    start_time = time.monotonic()
    progress_handle = progress_path.open("a", encoding="utf-8", buffering=1)

    def checkpoint_record(record: dict) -> None:
        nonlocal processed_count, generated_count, failed_count
        record_map[record["sample_id"]] = record
        progress_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        processed_count += 1
        if record.get("status") == "failed":
            failed_count += 1
        elif record.get("status") == "generated":
            generated_count += 1
        if processed_count % 100 == 0:
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
        if processed_count % 500 == 0 or processed_count == len(samples):
            elapsed = max(time.monotonic() - start_time, 1e-6)
            new_done = generated_count + failed_count
            rate = new_done / elapsed if new_done else 0.0
            eta = (len(samples) - processed_count) / rate if rate else None
            print(
                f"prepared {processed_count}/{len(samples)} images "
                f"(reused={processed_count - new_done}, generated={generated_count}, "
                f"failed={failed_count}, speed={rate:.2f} images/s, ETA={_format_duration(eta)})"
            )

    try:
        for record in recovered_without_log:
            progress_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if recovered_without_log:
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
        if pending:
            if prepare_workers == 1:
                _initialize_prepare_worker(settings)
                results = map(_process_image_task, pending)
                for result in results:
                    checkpoint_record(result)
            else:
                executor = ProcessPoolExecutor(
                    max_workers=prepare_workers,
                    initializer=_initialize_prepare_worker,
                    initargs=(settings,),
                )
                try:
                    for result in executor.map(_process_image_task, pending, chunksize=prepare_chunksize):
                        checkpoint_record(result)
                except BaseException:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)
    except KeyboardInterrupt:
        progress_handle.flush()
        os.fsync(progress_handle.fileno())
        export_json(
            resume_path,
            {
                "signature": signature,
                "complete": False,
                "processed": len(record_map),
                "total": len(samples),
                "message": "Interrupted safely; rerun the identical command to resume.",
            },
        )
        print(f"interrupted safely at {len(record_map)}/{len(samples)}; rerun the identical command")
        raise
    finally:
        progress_handle.close()

    records = [record_map[sample_id] for sample_id in ordered_ids if sample_id in record_map]
    failures = [record for record in records if record.get("status") == "failed"]
    per_split_classes: dict[str, Counter[str]] = defaultdict(Counter)
    split_groups: dict[str, set[str]] = defaultdict(set)
    bearing_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("status") in {"generated", "reused_valid_image"}:
            per_split_classes[record["split"]][record["label"]] += 1
            split_groups[record["split"]].add(record["split_group_id"])
            bearing_groups[record["split"]].add(record["group_id"])

    manifest = {
        "signature": signature,
        "complete": not failures and len(records) == len(samples),
        "condition": condition,
        "runtime": runtime_metadata(),
        "counts": {split: dict(counts) for split, counts in per_split_classes.items()},
        "groups": {split: sorted(values) for split, values in split_groups.items()},
        "bearing_groups": {split: sorted(values) for split, values in bearing_groups.items()},
        "records": records,
        "failure_count": len(failures),
    }
    if diagnostic_prior_metadata is not None:
        manifest["diagnostic_prior"] = diagnostic_prior_metadata
    export_json(manifest_path, manifest)
    export_json(state_path, {"signature": signature, "condition": condition, "complete": manifest["complete"]})
    if resume_path.exists() and manifest["complete"]:
        resume_path.unlink()
    if failures or not manifest["complete"]:
        raise RuntimeError(f"{len(failures)} image generations failed; see {manifest_path}")
    return output_root


def run_condition(args, cfg: dict) -> dict:
    sampler = getattr(args, "sampler", "none")
    class_weighting = getattr(args, "class_weighting", "none")
    if sampler != "none" and class_weighting != "none":
        raise ValueError("--sampler and --class-weighting are mutually exclusive (single-factor protocol)")
    test_groups = tuple(
        value.strip() for value in getattr(args, "test_groups", "").split(",") if value.strip()
    )
    val_groups = tuple(
        value.strip() for value in getattr(args, "val_groups", "").split(",") if value.strip()
    )
    primary_artifact_root, artifact_roots = _normalized_artifact_roots(
        getattr(args, "artifact_root", None),
        tuple(getattr(args, "reuse_artifact_roots", ()) or ()),
    )
    data_root = prepare_dataset(
        cfg,
        args.dataset,
        args.method,
        args.ratio,
        args.representation,
        args.seed,
        no_cs=args.no_cs,
        max_samples_per_group=args.max_samples_per_group,
        prepare_workers=getattr(args, "prepare_workers", 1),
        prepare_chunksize=getattr(args, "prepare_chunksize", 8),
        sparsity_k=getattr(args, "sparsity_k", None),
        gradient_lambda=getattr(args, "gradient_lambda", 0.25),
        diagnostic_prior_mu=getattr(args, "diagnostic_prior_mu", 0.5),
        diagnostic_prior_max_per_class=getattr(args, "diagnostic_prior_max_per_class", 4096),
        sparse_basis=getattr(args, "sparse_basis", "dct"),
        wavelet=getattr(args, "wavelet", None),
        cwt_scales_start=getattr(args, "cwt_scales_start", 1),
        cwt_scales_end=getattr(args, "cwt_scales_end", 128),
        stft_nperseg=getattr(args, "stft_nperseg", 64),
        stft_noverlap=getattr(args, "stft_noverlap", None),
        noise_snr_db=getattr(args, "noise_snr_db", None),
        train_conditions=tuple(getattr(args, "train_conditions", ()) or ()),
        test_condition=getattr(args, "test_condition", None),
        compression_baseline=getattr(args, "compression_baseline", None),
        pca_fit_samples=getattr(args, "pca_fit_samples", 4096),
        test_groups=test_groups,
        val_groups=val_groups,
        split_unit=getattr(args, "split_unit", "bearing"),
        artifact_root=primary_artifact_root,
        reuse_artifact_roots=artifact_roots[1:],
    )
    manifest = json.loads((data_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    training_config = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "freeze_ratio": args.freeze_ratio,
        "seed": args.seed,
    }
    if getattr(args, "amp", False):
        training_config["amp"] = True
    if class_weighting != "none":
        training_config["class_weighting"] = class_weighting
    if sampler != "none":
        training_config["balanced_sampling"] = sampler
    if getattr(args, "report_train_metrics", False):
        training_config["report_train_metrics"] = True
    checkpoint_selection = getattr(args, "checkpoint_selection", "best-val")
    evaluation_scope = getattr(args, "evaluation_scope", "formal-test")
    if evaluation_scope not in {"validation-only", "formal-test"}:
        raise ValueError("evaluation_scope must be 'validation-only' or 'formal-test'")
    if checkpoint_selection != "best-val":
        training_config["checkpoint_selection"] = checkpoint_selection
    if evaluation_scope != "formal-test":
        training_config["evaluation_scope"] = evaluation_scope
    if getattr(args, "fast_training", False):
        training_config["performance"] = {
            "fast_training": True,
            "gpu_image_preprocessing": True,
            "channels_last": True,
            "prefetch_factor": getattr(args, "prefetch_factor", 4),
            "eval_num_workers": getattr(args, "eval_num_workers", 0),
        }
    training_signature = canonical_hash(training_config)
    selected_artifact_root = _artifact_root_for_dataset(data_root, artifact_roots)
    result_root = (
        selected_artifact_root / "results" / args.dataset / manifest["signature"]
        / args.backbone / training_signature
    )
    if args.prepare_only:
        return {
            "dataset_root": str(data_root),
            "signature": manifest["signature"],
            "seed": args.seed,
            "prepared_only": True,
        }
    result = train_classifier(
        data_root,
        result_root,
        backbone=args.backbone,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        freeze_ratio=args.freeze_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=getattr(args, "amp", False),
        fast_training=getattr(args, "fast_training", False),
        prefetch_factor=getattr(args, "prefetch_factor", 4),
        eval_num_workers=getattr(args, "eval_num_workers", 0),
        class_weighting=class_weighting,
        balanced_sampling=sampler,
        report_train_metrics=getattr(args, "report_train_metrics", False),
        checkpoint_selection=checkpoint_selection,
        evaluation_scope=evaluation_scope,
    )
    summary = {
        "dataset_root": str(data_root),
        "result_root": str(result_root),
        "signature": manifest["signature"],
        "training_signature": training_signature,
        "seed": args.seed,
        "checkpoint": result["checkpoint"],
        "evaluation_scope": evaluation_scope,
        "independent_test_opened": bool(
            result.get("independent_test_opened", evaluation_scope == "formal-test" and "paper_metrics" in result)
        ),
    }
    if evaluation_scope == "formal-test":
        summary["paper_metrics"] = result["paper_metrics"]
    else:
        summary["validation_metrics"] = result["validation_metrics"]
        summary["formal_claim_authorized"] = False
    if "train_metrics" in result:
        summary["train_metrics"] = result["train_metrics"]
    summary_name = (
        "condition_summary.json"
        if evaluation_scope == "formal-test"
        else "condition_validation_summary.json"
    )
    export_json(result_root / summary_name, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=ROOT / "configs" / "default_config.yaml")
    cli.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "physical artifacts_v2 root for newly materialized datasets/cache/results; "
            "storage location does not enter scientific signatures"
        ),
    )
    cli.add_argument(
        "--reuse-artifact-root",
        type=Path,
        action="append",
        default=[],
        help="additional existing artifacts_v2 root searched before materializing a new condition",
    )
    cli.add_argument("--dataset", choices=("pu", "cwru"), default="pu")
    cli.add_argument("--method", choices=("omp", "lsomp", "ilsomp", "dp_ilsomp"), default="ilsomp")
    cli.add_argument("--ratio", type=float, default=0.5)
    cli.add_argument("--representation", choices=("cwt", "stft"), default="cwt")
    cli.add_argument("--backbone", default="resnet34_ema")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--epochs", type=int, default=30)
    cli.add_argument("--learning-rate", type=float, default=1e-3)
    cli.add_argument("--batch-size", type=int, default=32)
    cli.add_argument("--freeze-ratio", type=float, default=0.5)
    cli.add_argument("--num-workers", type=int, default=0)
    cli.add_argument("--amp", action="store_true", help="CUDA mixed precision training; keep consistent across a study")
    cli.add_argument(
        "--class-weighting",
        choices=("none", "balanced"),
        default="none",
        help="diagnostic loss weighting; balanced computes inverse-frequency weights from train only",
    )
    cli.add_argument(
        "--sampler",
        choices=("none", "class-balanced"),
        default="none",
        help="diagnostic balanced sampling; class-balanced draws samples with inverse-frequency class weights (classes balanced in expectation per epoch, not guaranteed per batch), weights from train only; mutually exclusive with --class-weighting",
    )
    cli.add_argument(
        "--test-groups",
        default="",
        help="comma-separated bearing codes held out as the test split (explicit holdout; overrides the random group split and enters the dataset signature)",
    )
    cli.add_argument(
        "--val-groups",
        default="",
        help="comma-separated bearing codes held out as the val split (model selection only; may omit classes, e.g. ball, which is recorded in the manifest)",
    )
    cli.add_argument(
        "--split-unit",
        choices=("bearing", "source-file", "window"),
        default="bearing",
        help=(
            "independent unit for the random split; source-file keeps every window from one "
            "raw recording in exactly one split for calibrated-bearing monitoring; window is "
            "a deliberately leaky diagnostic control and is never paper-valid performance"
        ),
    )
    cli.add_argument(
        "--report-train-metrics",
        action="store_true",
        help="diagnostic only: also evaluate the final selected model on the training split",
    )
    cli.add_argument(
        "--checkpoint-selection",
        choices=("best-val", "last"),
        default="best-val",
        help="model used for final test evaluation; use 'last' when validation omits a study class",
    )
    cli.add_argument(
        "--evaluation-scope",
        choices=("validation-only", "formal-test"),
        default="formal-test",
        help=(
            "validation-only never constructs or evaluates an independent-test DataLoader; "
            "formal-test preserves the paper-metric behaviour"
        ),
    )
    cli.add_argument(
        "--fast-training",
        action="store_true",
        help="future-run acceleration: compact uint8 loading, CUDA normalization and channels-last",
    )
    cli.add_argument("--prefetch-factor", type=int, default=4)
    cli.add_argument(
        "--eval-num-workers",
        type=int,
        default=0,
        help="validation/test workers in fast mode; 0 avoids persistent Windows worker memory",
    )
    cli.add_argument(
        "--prepare-workers",
        type=int,
        default=1,
        help="parallel processes for CS+CWT/STFT preparation; independent of DataLoader workers",
    )
    cli.add_argument("--prepare-chunksize", type=int, default=8)
    cli.add_argument("--no-cs", action="store_true")
    cli.add_argument("--sparsity-k", type=int, help="override compressed-sensing sparsity K")
    cli.add_argument("--gradient-lambda", type=float, default=0.25)
    cli.add_argument(
        "--diagnostic-prior-mu",
        type=float,
        default=0.5,
        help="DP-I-LsOMP Fisher-prior strength; used only by --method dp_ilsomp",
    )
    cli.add_argument(
        "--diagnostic-prior-max-per-class",
        type=int,
        default=4096,
        help="deterministic training-only sample cap per class for fitting the Fisher prior",
    )
    cli.add_argument("--sparse-basis", choices=("dct", "db4", "db8"), default="dct")
    cli.add_argument("--wavelet", help="CWT mother wavelet; default comes from the YAML config")
    cli.add_argument("--cwt-scales-start", type=int, default=1)
    cli.add_argument("--cwt-scales-end", type=int, default=128)
    cli.add_argument("--stft-nperseg", type=int, default=64)
    cli.add_argument("--stft-noverlap", type=int)
    cli.add_argument("--noise-snr-db", type=float, help="add deterministic Gaussian noise to test only")
    cli.add_argument(
        "--train-conditions",
        default="",
        help="comma-separated Paderborn source conditions for cross-condition training/validation",
    )
    cli.add_argument("--test-condition", help="held-out Paderborn target condition")
    cli.add_argument(
        "--compression-baseline",
        choices=("downsample", "pca", "random_projection"),
    )
    cli.add_argument("--pca-fit-samples", type=int, default=4096)
    cli.add_argument("--prepare-only", action="store_true")
    cli.add_argument("--max-samples-per-group", type=int)
    return cli


def main() -> None:
    args = parser().parse_args()
    args.reuse_artifact_roots = tuple(args.reuse_artifact_root)
    args.train_conditions = tuple(value.strip() for value in args.train_conditions.split(",") if value.strip())
    if not 0 < args.ratio <= 1:
        raise SystemExit("--ratio must be in (0, 1]")
    result = run_condition(args, load_config(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
