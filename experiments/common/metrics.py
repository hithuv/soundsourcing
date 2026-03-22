"""Separation metrics used in evaluation."""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    noise = reference - estimate
    return float(10.0 * np.log10((np.sum(reference**2) + EPS) / (np.sum(noise**2) + EPS)))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    projection = alpha * ref
    noise = est - projection
    return float(10.0 * np.log10((np.sum(projection**2) + EPS) / (np.sum(noise**2) + EPS)))


def sir(reference: np.ndarray, estimate: np.ndarray, mixture: np.ndarray) -> float:
    interference = mixture - reference
    residual = estimate - reference
    return float(10.0 * np.log10((np.sum(reference**2) + EPS) / (np.sum((residual + interference) ** 2) + EPS)))

