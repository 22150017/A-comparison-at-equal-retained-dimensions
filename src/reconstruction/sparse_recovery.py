





from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pywt
from scipy.fftpack import dct, idct


@dataclass
class ReconstructionResult:
    name: str
    reconstructed_signal: np.ndarray
    compression_ratio: float
    reconstruction_error: float
    snr_db: float


def dct_basis_transform(signal: np.ndarray) -> np.ndarray:
    
    return dct(signal, norm="ortho")


def idct_basis_transform(coeffs: np.ndarray) -> np.ndarray:
    
    return idct(coeffs, norm="ortho")


_basis_matrix_cache: Dict[Tuple[int, str], np.ndarray] = {}


def _normalize_sparse_basis(name: str) -> str:
    normalized = name.lower()
    if normalized == "dct":
        return normalized
    wavelet = pywt.Wavelet(normalized)
    if not wavelet.orthogonal:
        raise ValueError(f"sparse basis must be orthogonal: {name}")
    return wavelet.name


def _wavelet_synthesis_matrix(length: int, basis: str) -> np.ndarray:
    
    key = (length, basis)
    if key not in _basis_matrix_cache:
        identity = np.eye(length, dtype=np.float64)
        coefficients = pywt.wavedec(identity, basis, mode="periodization", axis=1)
        synthesis = np.concatenate(coefficients, axis=1)
        if synthesis.shape != (length, length):
            raise RuntimeError(f"unexpected {basis} basis shape: {synthesis.shape}")
        if not np.allclose(synthesis.T @ synthesis, np.eye(length), atol=1e-8):
            raise RuntimeError(f"{basis} periodized basis is not numerically orthonormal")
        _basis_matrix_cache[key] = synthesis
    return _basis_matrix_cache[key]


def _basis_forward(signal: np.ndarray) -> np.ndarray:
    if _sparse_basis == "dct":
        return dct_basis_transform(signal)
    synthesis = _wavelet_synthesis_matrix(len(signal), _sparse_basis)
    return synthesis.T @ signal


def _basis_inverse(coefficients: np.ndarray) -> np.ndarray:
    if _sparse_basis == "dct":
        return idct_basis_transform(coefficients)
    synthesis = _wavelet_synthesis_matrix(len(coefficients), _sparse_basis)
    return synthesis @ coefficients


def bernoulli_measurement_matrix(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    
    if m <= 0 or n <= 0:
        raise ValueError("measurement matrix dimensions must be positive")
    return rng.choice([-1.0, 1.0], size=(m, n)) / np.sqrt(m)


def compress_signal(signal: np.ndarray, compression_ratio: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    



    if not 0 < compression_ratio <= 1:
        raise ValueError("compression_ratio must be in (0, 1]")
    n = signal.shape[0]
    m = max(1, min(n, int(round(n * compression_ratio))))
    rng = np.random.default_rng(seed)
    phi = bernoulli_measurement_matrix(m, n, rng)
    y = phi @ signal
    return phi, y


def hard_threshold(coeffs: np.ndarray, k: int) -> np.ndarray:
    
    out = np.zeros_like(coeffs)
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if k == 0:
        return out
    idx = np.argsort(np.abs(coeffs))[-k:]
    out[idx] = coeffs[idx]
    return out


_theta_cache: Dict[Tuple[int, int, int, str], np.ndarray] = {}


_cs_seed: int = 42
_sparse_basis: str = "dct"
_gradient_lambda: float = 0.25
_diagnostic_atom_weights: Optional[np.ndarray] = None
_diagnostic_prior_mu: float = 0.5


def set_cs_seed(seed: int) -> None:
    
    global _cs_seed
    _cs_seed = seed
    clear_theta_cache()


def get_cs_seed() -> int:
    
    return _cs_seed


def set_sparse_basis(name: str) -> None:
    
    global _sparse_basis
    _sparse_basis = _normalize_sparse_basis(name)
    clear_theta_cache()


def get_sparse_basis() -> str:
    return _sparse_basis


def set_gradient_lambda(value: float) -> None:
    
    if not np.isfinite(value) or value < 0:
        raise ValueError("gradient lambda must be non-negative")
    global _gradient_lambda
    _gradient_lambda = float(value)


def get_gradient_lambda() -> float:
    return _gradient_lambda


def set_diagnostic_prior(weights: Optional[np.ndarray], mu: float = 0.5) -> None:
    
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("diagnostic prior mu must be non-negative")
    normalized = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if normalized is not None and (not np.all(np.isfinite(normalized)) or np.any(normalized < 0)):
        raise ValueError("diagnostic atom weights must be finite and non-negative")
    global _diagnostic_atom_weights, _diagnostic_prior_mu
    _diagnostic_atom_weights = normalized
    _diagnostic_prior_mu = float(mu)


def get_diagnostic_prior() -> tuple[Optional[np.ndarray], float]:
    weights = None if _diagnostic_atom_weights is None else _diagnostic_atom_weights.copy()
    return weights, _diagnostic_prior_mu


def _get_sensing_matrix(M: int, N: int, seed: int, basis: str | None = None) -> np.ndarray:
    








    selected_basis = _normalize_sparse_basis(basis or _sparse_basis)
    key = (M, N, seed, selected_basis)
    if key not in _theta_cache:
        rng = np.random.default_rng(seed)
        Phi = bernoulli_measurement_matrix(M, N, rng)
        if selected_basis == "dct":
            
            Theta = dct(Phi, axis=1, norm="ortho")
        else:
            Theta = Phi @ _wavelet_synthesis_matrix(N, selected_basis)
        _theta_cache[key] = Theta
        print(
            f"    [CS] cached sensing matrix Theta in R^{{{M}x{N}}} "
            f"(seed={seed}, basis={selected_basis})"
        )
    return _theta_cache[key]


def clear_theta_cache():
    
    _theta_cache.clear()


def _direct_dct_threshold(signal: np.ndarray, sparsity_k: int,
                          gradient_filter: bool = False) -> np.ndarray:
    
    alpha = _basis_forward(signal)
    if gradient_filter:
        gradient = np.abs(np.gradient(alpha))
        score = np.abs(alpha) + 0.25 * gradient
        idx = np.argsort(score)[-sparsity_k:]
        sparse = np.zeros_like(alpha)
        sparse[idx] = alpha[idx]
    else:
        sparse = hard_threshold(alpha, sparsity_k)
    return _basis_inverse(sparse)


def omp_reconstruct(signal: np.ndarray, compression_ratio: float,
                    sparsity_k: int) -> ReconstructionResult:
    




    if not 0 < compression_ratio <= 1:
        raise ValueError("compression_ratio must be in (0, 1]")
    N = signal.shape[0]
    M = max(1, min(N, int(round(N * compression_ratio))))

    
    alpha = _basis_forward(signal)
    rng = np.random.default_rng(_cs_seed)
    Phi = bernoulli_measurement_matrix(M, N, rng)
    y = Phi @ signal

    
    Theta = _get_sensing_matrix(M, N, _cs_seed)

    
    residual = y.copy()
    support = []
    max_iter = min(sparsity_k, M - 1)

    for _ in range(max_iter):
        
        corr = Theta.T @ residual              
        j = int(np.argmax(np.abs(corr)))
        if j in support:
            break
        support.append(j)

        
        Theta_S = Theta[:, support]            
        alpha_s, *_ = np.linalg.lstsq(Theta_S, y, rcond=None)

        
        residual = y - Theta_S @ alpha_s

        
        if np.linalg.norm(residual) < 1e-8 * np.linalg.norm(y):
            break

    
    alpha_hat = np.zeros(N)
    for idx_s, j in enumerate(support):
        alpha_hat[j] = alpha_s[idx_s]
    recon = _basis_inverse(alpha_hat)

    return build_result("omp", signal, recon, compression_ratio)


def lsomp_reconstruct(signal: np.ndarray, compression_ratio: float,
                      sparsity_k: int) -> ReconstructionResult:
    




    if not 0 < compression_ratio <= 1:
        raise ValueError("compression_ratio must be in (0, 1]")
    N = signal.shape[0]
    M = max(1, min(N, int(round(N * compression_ratio))))

    
    alpha = _basis_forward(signal)
    rng = np.random.default_rng(_cs_seed)
    Phi = bernoulli_measurement_matrix(M, N, rng)
    y = Phi @ signal
    Theta = _get_sensing_matrix(M, N, _cs_seed)

    
    extended_k = min(sparsity_k + 8, M - 1)
    residual = y.copy()
    support = []

    for _ in range(extended_k):
        corr = Theta.T @ residual
        j = int(np.argmax(np.abs(corr)))
        if j in support:
            break
        support.append(j)

        Theta_S = Theta[:, support]
        alpha_s, *_ = np.linalg.lstsq(Theta_S, y, rcond=None)
        residual = y - Theta_S @ alpha_s

        if np.linalg.norm(residual) < 1e-8 * np.linalg.norm(y):
            break

    
    alpha_hat = np.zeros(N)
    if len(support) > sparsity_k and len(alpha_s) == len(support):
        abs_vals = [(abs(alpha_s[i]), support[i]) for i in range(len(support))]
        abs_vals.sort(reverse=True)
        final_support = [item[1] for item in abs_vals[: min(sparsity_k, len(abs_vals))]]
        
        
        final_coeffs, *_ = np.linalg.lstsq(Theta[:, final_support], y, rcond=None)
        for j, coefficient in zip(final_support, final_coeffs):
            alpha_hat[j] = coefficient
    else:
        for idx_s, j in enumerate(support):
            alpha_hat[j] = alpha_s[idx_s]

    recon = _basis_inverse(alpha_hat)
    return build_result("lsomp", signal, recon, compression_ratio)


def ilsomp_reconstruct(signal: np.ndarray, compression_ratio: float,
                       sparsity_k: int) -> ReconstructionResult:
    





    if not 0 < compression_ratio <= 1:
        raise ValueError("compression_ratio must be in (0, 1]")
    N = signal.shape[0]
    M = max(1, min(N, int(round(N * compression_ratio))))

    
    alpha = _basis_forward(signal)
    rng = np.random.default_rng(_cs_seed)
    Phi = bernoulli_measurement_matrix(M, N, rng)
    y = Phi @ signal
    Theta = _get_sensing_matrix(M, N, _cs_seed)

    
    lam = _gradient_lambda
    residual = y.copy()
    support = []
    prev_abs_corr = np.zeros(N)
    max_iter = min(sparsity_k, M - 1)

    for k_iter in range(max_iter):
        
        corr = Theta.T @ residual        
        abs_corr = np.abs(corr)

        
        if k_iter > 0:
            delta_g = abs_corr - prev_abs_corr
            gradient_bonus = np.maximum(delta_g, 0)
            score = abs_corr + lam * gradient_bonus
        else:
            score = abs_corr.copy()

        
        for s in support:
            score[s] = -1.0

        
        j = int(np.argmax(score))
        support.append(j)

        
        Theta_S = Theta[:, support]
        alpha_s, *_ = np.linalg.lstsq(Theta_S, y, rcond=None)
        residual = y - Theta_S @ alpha_s

        
        prev_abs_corr = abs_corr.copy()

        
        if np.linalg.norm(residual) < 1e-8 * np.linalg.norm(y):
            break

    
    alpha_hat = np.zeros(N)
    for idx_s, j in enumerate(support):
        alpha_hat[j] = alpha_s[idx_s]
    recon = _basis_inverse(alpha_hat)

    return build_result("ilsomp", signal, recon, compression_ratio)


def dp_ilsomp_reconstruct(signal: np.ndarray, compression_ratio: float,
                          sparsity_k: int) -> ReconstructionResult:
    





    if not 0 < compression_ratio <= 1:
        raise ValueError("compression_ratio must be in (0, 1]")
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.shape[0]
    m = max(1, min(n, int(round(n * compression_ratio))))
    weights = _diagnostic_atom_weights
    if weights is None:
        raise RuntimeError("DP-I-LsOMP requires a fitted training-only diagnostic prior")
    if weights.shape != (n,):
        raise ValueError(f"diagnostic prior length {len(weights)} does not match signal length {n}")

    rng = np.random.default_rng(_cs_seed)
    phi = bernoulli_measurement_matrix(m, n, rng)
    y = phi @ signal
    theta = _get_sensing_matrix(m, n, _cs_seed)
    residual = y.copy()
    support: list[int] = []
    prev_abs_corr = np.zeros(n)
    max_iter = min(sparsity_k, m - 1)
    alpha_s = np.empty(0, dtype=np.float64)
    for iteration in range(max_iter):
        abs_corr = np.abs(theta.T @ residual)
        score = abs_corr * (1.0 + _diagnostic_prior_mu * weights)
        if iteration > 0:
            score += _gradient_lambda * np.maximum(abs_corr - prev_abs_corr, 0.0)
        if support:
            score[np.asarray(support, dtype=np.int64)] = -1.0
        support.append(int(np.argmax(score)))
        theta_support = theta[:, support]
        alpha_s, *_ = np.linalg.lstsq(theta_support, y, rcond=None)
        residual = y - theta_support @ alpha_s
        prev_abs_corr = abs_corr
        if np.linalg.norm(residual) < 1e-8 * np.linalg.norm(y):
            break

    alpha_hat = np.zeros(n)
    for support_index, atom_index in enumerate(support):
        alpha_hat[atom_index] = alpha_s[support_index]
    recon = _basis_inverse(alpha_hat)
    return build_result("dp_ilsomp", signal, recon, compression_ratio)


def reconstruction_error(signal: np.ndarray, recon: np.ndarray) -> float:
    
    return float(np.linalg.norm(signal - recon) / (np.linalg.norm(signal) + 1e-12))


def snr_db(signal: np.ndarray, recon: np.ndarray) -> float:
    
    signal_power = np.sum(signal ** 2) + 1e-12
    noise_power = np.sum((signal - recon) ** 2) + 1e-12
    return float(10.0 * np.log10(signal_power / noise_power))


def build_result(name: str, signal: np.ndarray, recon: np.ndarray,
                 compression_ratio: float) -> ReconstructionResult:
    return ReconstructionResult(
        name=name,
        reconstructed_signal=recon,
        compression_ratio=compression_ratio,
        reconstruction_error=reconstruction_error(signal, recon),
        snr_db=snr_db(signal, recon),
    )


def get_reconstructors() -> Dict[str, Callable[[np.ndarray, float, int], ReconstructionResult]]:
    
    return {
        "omp": omp_reconstruct,
        "lsomp": lsomp_reconstruct,
        "ilsomp": ilsomp_reconstruct,
        "dp_ilsomp": dp_ilsomp_reconstruct,
    }
