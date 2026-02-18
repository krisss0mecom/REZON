#!/usr/bin/env python3
import numpy as np


def _entropy_from_hist(phi: np.ndarray, bins: int = 20) -> float:
    hist, _ = np.histogram(phi % (2.0 * np.pi), bins=bins, range=(0.0, 2.0 * np.pi), density=False)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def pairwise_phase_correlation(phi: np.ndarray) -> float:
    phi = np.asarray(phi, dtype=np.float64)
    n = phi.size
    if n < 2:
        return 0.0
    diffs = phi[:, None] - phi[None, :]
    return float(np.mean(np.cos(diffs)))


def mutual_information_phase(phi: np.ndarray, bins: int = 20) -> float:
    # Proxy entropy of marginal phase distribution.
    return _entropy_from_hist(np.asarray(phi, dtype=np.float64), bins=bins)


def bipartition_entropy(phi: np.ndarray, split: float = 0.5, bins: int = 20) -> float:
    phi = np.asarray(phi, dtype=np.float64)
    n = phi.size
    mid = int(max(1, min(n - 1, round(n * split))))
    ent_a = _entropy_from_hist(phi[:mid], bins=bins)
    ent_b = _entropy_from_hist(phi[mid:], bins=bins)
    return float(0.5 * (ent_a + ent_b))


def chsh_proxy(phi: np.ndarray) -> float:
    # CHSH-like classical proxy over phase-projected observables.
    phi = np.asarray(phi, dtype=np.float64)
    a1 = np.cos(phi)
    a2 = np.sin(phi)
    b1 = np.cos(phi + np.pi / 4.0)
    b2 = np.sin(phi + np.pi / 4.0)
    e11 = float(np.mean(a1 * b1))
    e12 = float(np.mean(a1 * b2))
    e21 = float(np.mean(a2 * b1))
    e22 = float(np.mean(a2 * b2))
    return float(abs(e11 + e12 + e21 - e22))


def compute_all_metrics(phi: np.ndarray) -> dict:
    phi = np.asarray(phi, dtype=np.float64)
    return {
        "pairwise_correlation": pairwise_phase_correlation(phi),
        "mutual_info": mutual_information_phase(phi),
        "bipartition_entropy": bipartition_entropy(phi),
        "chsh_proxy": chsh_proxy(phi),
    }


if __name__ == "__main__":
    np.random.seed(42)
    phi = np.random.uniform(0.0, 2.0 * np.pi, 8)
    print(compute_all_metrics(phi))
