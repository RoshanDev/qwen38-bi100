"""Torch-free Gate 5 metrics and stop-loss decision."""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


DECISION_STOP = "stop_mtp"
DECISION_ALIGN = "align_semantics"
DECISION_GPU_MAYBE = "gpu_k1_maybe"
DECISION_GPU_SPEC = "gpu_spec_path"

OFFICIAL_POSITION_OFFSETS = (0,)
DIAGNOSTIC_POSITION_OFFSETS = (-2, -1, 1, 2)


def softmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        raise ValueError("softmax on empty scores")
    peak = max(xs)
    exps = [math.exp(float(x) - peak) for x in xs]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(xs)] * len(xs)
    return [v / total for v in exps]


def attention_weights(query: Sequence[float], keys: Sequence[Sequence[float]], scale: float) -> list[float]:
    if not keys:
        raise ValueError("attention needs at least one key")
    if len(query) != len(keys[0]):
        raise ValueError("query/key dim mismatch")
    scores = [sum(float(q) * float(k) for q, k in zip(query, key)) * float(scale) for key in keys]
    return softmax(scores)


def rank_of(logits: Sequence[float], target_id: int) -> int:
    if target_id < 0 or target_id >= len(logits):
        raise ValueError(f"target_id {target_id} out of range {len(logits)}")
    target = float(logits[target_id])
    better = sum(1 for value in logits if float(value) > target)
    return better + 1


def topk_ids(logits: Sequence[float], k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    indexed = sorted(enumerate(logits), key=lambda item: (-float(item[1]), item[0]))
    return [idx for idx, _ in indexed[:k]]


def step_stats(logits: Sequence[float], target_id: int, top_k: int = 5) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    top_ids = topk_ids(logits, min(top_k, len(logits)))
    draft_id = int(top_ids[0])
    rank = rank_of(logits, int(target_id))
    draft_logit = float(logits[draft_id])
    target_logit = float(logits[int(target_id)])
    second = float(logits[top_ids[1]]) if len(top_ids) > 1 else draft_logit
    return {
        "draft_id": draft_id,
        "target_id": int(target_id),
        "top1_hit": draft_id == int(target_id),
        "top5_hit": int(target_id) in top_ids[: min(5, len(top_ids))],
        "rank": rank,
        "logit_margin_draft_minus_target": draft_logit - target_logit,
        "logit_margin_top1_minus_top2": draft_logit - second,
        "draft_logit": draft_logit,
        "target_logit": target_logit,
        "top_ids": top_ids,
    }


def summarize_steps(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    n = len(items)
    if n == 0:
        return {
            "n": 0,
            "top1_hits": 0,
            "top5_hits": 0,
            "top1_rate": 0.0,
            "top5_rate": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mean_margin_draft_minus_target": 0.0,
        }
    ranks = sorted(int(row["rank"]) for row in items)
    mid = n // 2
    median = ranks[mid] if n % 2 else 0.5 * (ranks[mid - 1] + ranks[mid])
    top1 = sum(1 for row in items if row["top1_hit"])
    top5 = sum(1 for row in items if row["top5_hit"])
    return {
        "n": n,
        "top1_hits": top1,
        "top5_hits": top5,
        "top1_rate": top1 / n,
        "top5_rate": top5 / n,
        "mean_rank": sum(ranks) / n,
        "median_rank": float(median),
        "mean_margin_draft_minus_target": sum(
            float(row["logit_margin_draft_minus_target"]) for row in items
        )
        / n,
    }


def decide_from_top1(rate: float) -> dict[str, Any]:
    pct = float(rate) * 100.0
    if rate < 0.30:
        decision = DECISION_STOP
        meaning = "停止 MTP 路线，继续保留实验分支"
    elif rate < 0.60:
        decision = DECISION_ALIGN
        meaning = "仍有 hidden/token/position/算子语义不一致，先与上游 logits 对齐"
    elif rate < 0.75:
        decision = DECISION_GPU_MAYBE
        meaning = "可以考虑 GPU K=1，但收益未必稳定"
    else:
        decision = DECISION_GPU_SPEC
        meaning = "才值得实现 GPU MTP + 一次 target forward 验两个 token"
    return {
        "top1_rate": float(rate),
        "top1_pct": pct,
        "decision": decision,
        "meaning": meaning,
        "band": (
            "<30%"
            if rate < 0.30
            else "30%-60%"
            if rate < 0.60
            else "60%-75%"
            if rate < 0.75
            else ">75%"
        ),
    }


def is_official_offset(offset: int) -> bool:
    return int(offset) in OFFICIAL_POSITION_OFFSETS
