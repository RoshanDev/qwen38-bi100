#!/usr/bin/env python3
"""Offline Gate 5 replay: recorded positions + optional one-layer MTP KV cache.

CPU only. Does not start the observer hook or a serving container.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mtp_k1.head import MTPHead
from mtp_k1.kv import MTPKVCache
from mtp_k1.replay_stats import (
    DIAGNOSTIC_POSITION_OFFSETS,
    decide_from_top1,
    step_stats,
    summarize_steps,
)


def _tensor_stats(logits: Any, target_id: int) -> dict[str, Any]:
    flat = logits.float().reshape(-1)
    values = flat.tolist()
    return step_stats(values, int(target_id), top_k=5)


def lm_head_pred(head: MTPHead, hidden: Any) -> int:
    import torch

    row = hidden
    if row.dim() == 1:
        row = row.view(1, -1)
    scores = torch.nn.functional.linear(row.float(), head.lm_head.float())
    return int(torch.argmax(scores.reshape(-1)).item())


def replay_pass(
    head: MTPHead,
    hiddens: Any,
    steps: list[dict[str, Any]],
    *,
    use_cache: bool,
    position_offset: int,
    limit: int | None = None,
) -> dict[str, Any]:
    cache = MTPKVCache() if use_cache else None
    rows: list[dict[str, Any]] = []
    identity_hits = 0
    chosen = steps if limit is None else steps[: int(limit)]
    started = time.perf_counter()
    for step in chosen:
        hidden = hiddens[int(step["hidden_index"])]
        pred = lm_head_pred(head, hidden)
        if pred == int(step["sampled_token"]):
            identity_hits += 1
        position = int(step["position"]) + int(position_offset)
        draft, logits = head.propose(
            int(step["sampled_token"]),
            hidden,
            position=position,
            cache=cache,
        )
        stats = _tensor_stats(logits, int(step["next_token"]))
        stats.update(
            {
                "index": int(step["index"]),
                "sampled_token": int(step["sampled_token"]),
                "next_token": int(step["next_token"]),
                "position_recorded": int(step["position"]),
                "position_used": position,
                "position_offset": int(position_offset),
                "source": step.get("source"),
                "lm_head_pred": pred,
                "lm_head_matches_sampled": pred == int(step["sampled_token"]),
                "draft_id": int(draft),
                "cache_len": 0 if cache is None else len(cache),
            }
        )
        rows.append(stats)
    summary = summarize_steps(rows)
    summary.update(
        {
            "use_cache": bool(use_cache),
            "position_offset": int(position_offset),
            "official": bool(use_cache) and int(position_offset) == 0,
            "lm_head_identity_rate": identity_hits / max(1, len(rows)),
            "elapsed_s": time.perf_counter() - started,
            "cache_final_len": 0 if cache is None else len(cache),
            "steps": rows,
        }
    )
    return summary


def compact_pass(result: dict[str, Any]) -> dict[str, Any]:
    keep = dict(result)
    keep.pop("steps", None)
    keep["sample"] = [
        {
            "index": row["index"],
            "draft_id": row["draft_id"],
            "next_token": row["next_token"],
            "rank": row["rank"],
            "top1_hit": row["top1_hit"],
            "position_used": row["position_used"],
        }
        for row in result.get("steps", [])[:4]
    ]
    return keep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--diagnostic-steps", type=int, default=32)
    args = parser.parse_args()

    import torch

    payload = torch.load(args.dump, map_location="cpu")
    steps = list(payload["steps"])
    if args.max_steps > 0:
        steps = steps[: int(args.max_steps)]
    if len(steps) < 32:
        print(f"WARN only {len(steps)} recorded steps", flush=True)
    hiddens = payload["hiddens"]
    head = MTPHead.load(args.model, device="cpu")
    official = replay_pass(head, hiddens, steps, use_cache=True, position_offset=0)
    control = replay_pass(head, hiddens, steps, use_cache=False, position_offset=0)
    diagnostics: list[dict[str, Any]] = []
    diag_limit = min(int(args.diagnostic_steps), len(steps))
    for offset in DIAGNOSTIC_POSITION_OFFSETS:
        diagnostics.append(
            compact_pass(
                replay_pass(
                    head,
                    hiddens,
                    steps,
                    use_cache=True,
                    position_offset=offset,
                    limit=diag_limit,
                )
            )
        )
    decision = decide_from_top1(float(official["top1_rate"]))
    report = {
        "gate": "5-kv-replay",
        "dump": args.dump,
        "n_steps": len(steps),
        "prompt": payload.get("prompt"),
        "prompt_len_tokenizer": payload.get("prompt_len_tokenizer"),
        "official_cache": compact_pass(official),
        "control_no_cache": compact_pass(control),
        "diagnostics_position_offsets": diagnostics,
        "decision": decision,
        "note": {
            "official": "cache=on, recorded position, no offset",
            "control": "cache=off 1x1 attention, recorded position",
            "diagnostics": "cache=on, position ±1/±2, first N steps only, not an official config",
            "metric": "draft top-1 hit rate against the next target greedy token; not speculative accepted-token rate",
        },
        "official_steps": official["steps"],
        "control_steps": control["steps"],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "gate": "5-kv-replay",
            "n_steps": len(steps),
            "official_top1": official["top1_rate"],
            "official_top5": official["top5_rate"],
            "official_mean_rank": official["mean_rank"],
            "control_top1": control["top1_rate"],
            "decision": decision,
        },
        ensure_ascii=False,
    ))
    print("MTP_GATE5_KV_REPLAY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
