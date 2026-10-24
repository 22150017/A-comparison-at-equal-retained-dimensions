from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.random.get_rng_state()
        state["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    try:
        import torch

        torch.random.set_rng_state(state["torch_cpu"].detach().cpu())
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all([value.detach().cpu() for value in state["torch_cuda"]])
    except ImportError:
        pass


def canonical_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    metadata["packages"] = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "torchvision", "Pillow", "PyWavelets", "thop", "pubfig"):
        try:
            metadata["packages"][package] = version(package)
        except PackageNotFoundError:
            metadata["packages"][package] = None
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        metadata["torch"] = None
    return metadata
