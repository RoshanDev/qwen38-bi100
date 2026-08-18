"""Assemble recorded target steps. Positions are taken from dumps, never inferred."""
from __future__ import annotations

from typing import Any


def classify_dump(rows: int, padded_cutoff: int = 8) -> str:
    if rows > 8:
        return "prefill" if rows <= 256 else "skip"
    if rows <= 0:
        return "skip"
    return "decode"


def assemble_records(token_ids: list[int], dumps: list[dict[str, Any]]) -> dict[str, Any]:
    if len(token_ids) < 2:
        raise ValueError("need at least two generated tokens to score MTP")
    usable = [row for row in dumps if row.get("kind") in {"prefill", "decode"}]
    if not usable:
        raise ValueError("no usable hidden dumps")
    steps: list[dict[str, Any]] = []
    hidden_index = 0
    for idx, dump in enumerate(usable):
        sampled = token_ids[idx] if idx < len(token_ids) else None
        nxt = token_ids[idx + 1] if idx + 1 < len(token_ids) else None
        if sampled is None or nxt is None:
            break
        position = dump.get("position")
        if position is None:
            raise ValueError(f"dump {idx} missing recorded position")
        steps.append(
            {
                "index": len(steps),
                "sampled_token": int(sampled),
                "next_token": int(nxt),
                "position": int(position),
                "source": "prefill_last" if dump.get("kind") == "prefill" else "decode",
                "hidden_rows": int(dump.get("rows") or 0),
                "dump_how": dump.get("how"),
                "hidden_index": hidden_index,
            }
        )
        hidden_index += 1
    if len(steps) < 2:
        raise ValueError(f"assembled only {len(steps)} steps")
    return {
        "n_tokens": len(token_ids),
        "n_dumps": len(dumps),
        "n_usable_dumps": len(usable),
        "n_steps": len(steps),
        "steps": steps,
    }
