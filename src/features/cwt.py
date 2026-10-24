





from __future__ import annotations

from pathlib import Path

import numpy as np
import pywt
from matplotlib import colormaps
from PIL import Image


def _build_jet_lut() -> np.ndarray:
    



    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0, 1)
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut



_VIRIDIS_LUT = (colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def normalize_wavelet_name(name: str) -> str:
    lower = name.lower()
    if lower in {"morlet", "morl"}:
        return "morl"
    return name


def signal_to_cwt_image(
    signal: np.ndarray,
    wavelet: str = "morl",
    scales: np.ndarray | None = None,
    image_size: int = 224,
) -> Image.Image:
    
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)) or image_size <= 0:
        raise ValueError("CWT requires a finite vector and positive image size")
    if scales is None:
        scales = np.arange(1, 129)
    if not np.all(np.isfinite(scales)) or np.any(np.asarray(scales) <= 0):
        raise ValueError("CWT scales must be finite and positive")

    
    coeffs, _ = pywt.cwt(signal, scales, normalize_wavelet_name(wavelet))
    magnitude = np.abs(coeffs)

    
    magnitude = magnitude / (magnitude.max() + 1e-12)

    
    
    idx = (magnitude * 255).astype(np.uint8)
    idx = idx[::-1, :]

    
    rgb = _VIRIDIS_LUT[idx]

    
    pil_image = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
    return pil_image


def save_cwt_image(signal: np.ndarray, output_path: Path, wavelet: str = "morl", image_size: int = 224, scales: np.ndarray | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = signal_to_cwt_image(signal, wavelet=wavelet, image_size=image_size, scales=scales)
    image.save(output_path)
