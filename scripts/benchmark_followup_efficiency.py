





from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_clean_study import _stft_image, load_config, load_samples
from src.models.attention_resnet import build_backbone
from src.reconstruction.compression_baselines import (
    downsample_interpolate,
    fit_pca_baseline,
    random_projection_reconstruct,
)
from src.reconstruction.sparse_recovery import (
    get_reconstructors,
    set_cs_seed,
    set_gradient_lambda,
    set_sparse_basis,
)
from src.utils.io_tools import export_json, stratified_group_split_three_way
from src.utils.reproducibility import canonical_hash, file_sha256

DEFAULT_PROTOCOL = ROOT / "configs" / "followup_evidence_v1.json"


def _stats_ms(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float) * 1000.0
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "n": len(array),
    }


def _hardware_profile() -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _training_samples(cfg: dict, dataset: str, seed: int) -> list:
    samples = load_samples(cfg, dataset)
    labels = [sample.label for sample in samples]
    groups = [str(Path(sample.source_file).resolve()) for sample in samples]
    train, _, _ = stratified_group_split_three_way(
        labels, groups, cfg["training"]["train_ratio"], cfg["training"]["val_ratio"], seed
    )
    return [samples[int(index)] for index in train]


def _method_transform(method: str, ratio: float, seed: int, pca_model=None):
    if method == "no_cs":
        return lambda signal: np.asarray(signal, dtype=np.float64)
    if method == "ilsomp":
        set_cs_seed(seed); set_sparse_basis("dct"); set_gradient_lambda(0.25)
        reconstructor = get_reconstructors()["ilsomp"]
        return lambda signal: reconstructor(signal, ratio, 64).reconstructed_signal
    if method == "downsample":
        return lambda signal: downsample_interpolate(signal, ratio)
    if method == "pca":
        if pca_model is None:
            raise ValueError("PCA timing requires a fitted training-only model")
        return pca_model.reconstruct
    if method == "random_projection":
        return lambda signal: random_projection_reconstruct(signal, ratio, seed)
    raise ValueError(f"unknown efficiency method: {method}")


def _measure_method(method: str, signals: list[np.ndarray], repeats: int, warmup: int, ratio: float, seed: int, pca_model=None) -> dict:
    transform = _method_transform(method, ratio, seed, pca_model)
    for signal in signals[:warmup]:
        _stft_image(transform(signal), 224, 64, 32)
    transform_times: list[float] = []
    end_to_image_times: list[float] = []
    total = repeats * len(signals)
    completed = 0
    started_all = time.perf_counter()
    report_every = max(1, total // 10)
    for _ in range(repeats):
        for signal in signals:
            started = time.perf_counter(); processed = transform(signal); transformed = time.perf_counter()
            _stft_image(processed, 224, 64, 32); finished = time.perf_counter()
            transform_times.append(transformed - started)
            end_to_image_times.append(finished - started)
            completed += 1
            if completed % report_every == 0 or completed == total:
                elapsed = max(time.perf_counter() - started_all, 1e-9)
                speed = completed / elapsed
                eta = (total - completed) / speed if speed > 0 else 0.0
                print(
                    f"  [{method}] {completed}/{total}, speed={speed:.2f} samples/s, "
                    f"ETA={eta:.1f}s",
                    flush=True,
                )
    transform_stats = _stats_ms(transform_times)
    pipeline_stats = _stats_ms(end_to_image_times)
    return {
        "transform_latency_ms": transform_stats,
        "transform_plus_stft_latency_ms": pipeline_stats,
        "transform_throughput_samples_per_second": (
            float(1000.0 / transform_stats["mean"]) if transform_stats["mean"] > 0 else None
        ),
        "transform_plus_stft_throughput_samples_per_second": (
            float(1000.0 / pipeline_stats["mean"]) if pipeline_stats["mean"] > 0 else None
        ),
        "retention_ratio": None if method == "no_cs" else ratio,
        "measurement_count": 1024 if method == "no_cs" else int(round(1024 * ratio)),
        "offline_fit_excluded": method == "pca",
    }


def _classifier_profile(backbone: str, warmup: int = 30, runs: int = 200) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(backbone, num_classes=4, pretrained=False).to(device).eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    macs_gmacs = None
    macs_note = None
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        macs_gmacs = float(macs / 1e9)
    except Exception as exc:
        
        macs_note = f"THOP profiling unavailable: {type(exc).__name__}: {exc}"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup): model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
        values = []
        for _ in range(runs):
            started = time.perf_counter(); model(dummy)
            if device.type == "cuda": torch.cuda.synchronize()
            values.append(time.perf_counter() - started)
    return {
        "backbone": backbone,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "macs_gmacs": macs_gmacs,
        "macs_note": macs_note,
        "operation_count_definition": "THOP multiply-accumulate count divided by 1e9",
        "parameter_memory_mb_fp32": float(
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 2**20
        ),
        "peak_cuda_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None
        ),
        "latency_ms": _stats_ms(values),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    cli.add_argument("--config", type=Path, default=ROOT / "configs" / "default_config.yaml")
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    settings = protocol["efficiency"]
    hardware = _hardware_profile()
    signature_payload = {
        "followup_protocol": canonical_hash(protocol, length=24),
        "methods": protocol["efficiency_methods"], "ratio": 0.5,
        "settings": settings,
        "hardware": hardware,
        "implementation_sha256": file_sha256(Path(__file__)),
        "config_sha256": file_sha256(args.config),
    }
    signature = canonical_hash(signature_payload, length=24)
    output = ROOT / "artifacts_v2" / "studies" / "followup" / f"efficiency_{signature}_summary.json"
    summary = {"complete": False, "status": "initializing", "protocol_signature": signature, "protocol": signature_payload, "independent_test_opened": False, "results": {}}
    summary["test_signal_materialized_before_training_subset_filter"] = True
    summary["test_metrics_used"] = False
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("protocol_signature") != signature: raise RuntimeError("incompatible efficiency summary")
        summary = previous
    methods = list(protocol["efficiency_methods"])
    pending = [method for method in methods if method not in summary["results"]]
    if args.dry_run:
        print(json.dumps({
            "protocol_signature": signature,
            "summary": str(output),
            "methods": methods,
            "pending": pending,
            "timed_samples_per_method": int(settings["timed_samples"]),
            "repeats": int(settings["repeats"]),
            "timed_observations_per_method": int(settings["timed_samples"]) * int(settings["repeats"]),
            "hardware": hardware,
            "uses_training_split_only": True,
            "independent_test_will_open": False,
        }, indent=2))
        return
    cfg = load_config(args.config)
    samples = _training_samples(cfg, protocol["dataset"], int(protocol["development_seed"]))
    ordered = sorted(samples, key=lambda item: (str(item.source_file), int(item.segment_id or 0)))
    timed = ordered[:int(settings["timed_samples"])]
    signals = [np.asarray(sample.signal, dtype=np.float64) for sample in timed]
    fit_samples = ordered[:int(settings["pca_fit_samples"])]
    pca_fit_started = time.perf_counter()
    pca_model = fit_pca_baseline(
        [np.asarray(sample.signal, dtype=np.float64) for sample in fit_samples],
        [f"{sample.source_file}:{sample.segment_id}" for sample in fit_samples],
        0.5, int(protocol["development_seed"]), int(settings["pca_fit_samples"]),
    )
    pca_fit_seconds = time.perf_counter() - pca_fit_started
    for method in methods:
        if method in summary["results"]: continue
        print(f"[efficiency] method={method}")
        summary["results"][method] = _measure_method(
            method, signals, int(settings["repeats"]), int(settings["warmup_samples"]),
            0.5, int(protocol["development_seed"]), pca_model,
        )
        summary["status"] = "running"; export_json(output, summary)
    summary["classifier"] = _classifier_profile("resnet34_ema")
    summary["offline_fit"] = {
        "pca_fit_seconds": float(pca_fit_seconds),
        "pca_fit_sample_count": len(fit_samples),
    }
    summary["complete"] = set(summary["results"]) == set(methods)
    summary["status"] = "complete" if summary["complete"] else "incomplete"
    summary["claim_boundary"] = (
        "Device-specific engineering evidence. Per-sample transform latency is steady-state after "
        "warm-up. Cached Theta construction and cached random-projection inverse preparation occur before "
        "timing; ILSOMP per-window Phi generation and measurement multiplication remain timed. "
        "PCA offline fitting time is separately disclosed."
    )
    export_json(output, summary)
    print(json.dumps({"status": summary["status"], "summary": str(output)}, indent=2))


if __name__ == "__main__":
    main()
