"""Gemma RMSNorm used by Qwen3.5/3.8 adapters.

Verified semantics (do not change):

    rms = x / sqrt(mean(x^2) + eps)
    output = rms * (1.0 + weight)

This is NOT standard RMSNorm ``rms * weight``.
"""
from __future__ import annotations

from typing import Sequence


def gemma_rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = 1e-6) -> list[float]:
    if len(x) != len(weight):
        raise ValueError(f"x/weight length mismatch: {len(x)} vs {len(weight)}")
    mean_sq = sum(float(v) * float(v) for v in x) / float(len(x))
    inv = (mean_sq + float(eps)) ** -0.5
    return [float(v) * inv * (1.0 + float(w)) for v, w in zip(x, weight)]


def standard_rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = 1e-6) -> list[float]:
    if len(x) != len(weight):
        raise ValueError(f"x/weight length mismatch: {len(x)} vs {len(weight)}")
    mean_sq = sum(float(v) * float(v) for v in x) / float(len(x))
    inv = (mean_sq + float(eps)) ** -0.5
    return [float(v) * inv * float(w) for v, w in zip(x, weight)]
