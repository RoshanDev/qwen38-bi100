#!/usr/bin/env python3
"""Gate 3: T=0 greedy accept/reject for MTP K=1.

The token-identity rule is CPU-only. Live 8K provides the no-MTP target
token. Real MTP draft still needs 64-layer hidden (Gate 3 live-hidden).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Sequence


def accept_reject_greedy(
    draft_ids: Sequence[int],
    target_ids: Sequence[int],
) -> dict[str, object]:
    """Accept the longest matching prefix. K=1 uses length-1 sequences."""
    if not draft_ids:
        raise ValueError("draft_ids must not be empty")
    n = min(len(draft_ids), len(target_ids))
    accepted = 0
    for i in range(n):
        if int(draft_ids[i]) != int(target_ids[i]):
            break
        accepted += 1
    return {
        "draft_ids": [int(x) for x in draft_ids],
        "target_ids": [int(x) for x in target_ids],
        "accepted": accepted,
        "bonus_target": accepted < len(target_ids),
        "token_identical_to_target_prefix": accepted == n and list(draft_ids[:n]) == list(target_ids[:n]),
    }


def k1_step(draft_id: int, target_id: int) -> dict[str, object]:
    result = accept_reject_greedy([draft_id], [target_id])
    result["next_token"] = draft_id if result["accepted"] == 1 else target_id
    result["accepted_draft"] = result["accepted"] == 1
    return result


def k1_rounds(draft_ids: Sequence[int], target_ids: Sequence[int]) -> dict[str, object]:
    """Sequential K=1 state: each draft is checked against the next target token."""
    if not draft_ids or not target_ids:
        raise ValueError("draft_ids and target_ids must not be empty")
    steps: list[dict[str, object]] = []
    emitted: list[int] = []
    for i, draft_id in enumerate(draft_ids):
        if i >= len(target_ids):
            break
        step = k1_step(int(draft_id), int(target_ids[i]))
        step["round"] = i
        steps.append(step)
        emitted.append(int(step["next_token"]))
        if not step["accepted_draft"]:
            # After reject, remaining drafts are invalid; emit leftover target.
            emitted.extend(int(x) for x in target_ids[i + 1 :])
            break
    return {
        "rounds": steps,
        "emitted_ids": emitted,
        "matches_target": emitted == [int(x) for x in target_ids[: len(emitted)]],
        "all_accepted": all(bool(s["accepted_draft"]) for s in steps),
    }


def k1_pipeline(t0: int, draft_ids: Sequence[int], target_ids: Sequence[int]) -> dict[str, object]:
    """T=0 first token is always the target t0; later tokens go through K=1 verify."""
    body = k1_rounds(draft_ids, target_ids)
    emitted = [int(t0)] + [int(x) for x in body["emitted_ids"]]
    no_mtp = [int(t0)] + [int(x) for x in target_ids]
    return {
        "t0": int(t0),
        "rounds": body["rounds"],
        "emitted_ids": emitted,
        "t0_identity": emitted[0] == int(t0),
        "matches_no_mtp_prefix": emitted == no_mtp[: len(emitted)],
        "all_accepted": body["all_accepted"],
    }


def probe_greedy_token(base_url: str, prompt: str, timeout: int = 60) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": "Qwen3.8-27B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = data["choices"][0]
    token = ((choice.get("logprobs") or {}).get("content") or [{}])[0]
    return {
        "text": choice.get("message", {}).get("content"),
        "token": token.get("token"),
        "logprob": token.get("logprob"),
        "top_logprobs": token.get("top_logprobs") or [],
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-url", default="")
    parser.add_argument("--prompt", default="只回复字母 A，不要其它内容。")
    parser.add_argument("--gate2", default="")
    parser.add_argument("--draft-id", type=int, default=-1)
    parser.add_argument("--target-id", type=int, default=-1)
    parser.add_argument("--target-token", default="")
    args = parser.parse_args()

    probe: dict[str, Any] | None = None
    if args.target_url:
        probe = probe_greedy_token(args.target_url, args.prompt)
        if not args.target_token:
            args.target_token = str(probe.get("token") or probe.get("text") or "")

    draft_id = int(args.draft_id)
    if draft_id < 0 and args.gate2:
        gate2 = json.loads(Path(args.gate2).read_text(encoding="utf-8"))
        draft_id = int(gate2["top1_id"])

    target_id = int(args.target_id)
    step = None
    rounds = None
    if draft_id >= 0 and target_id >= 0:
        step = k1_step(draft_id, target_id)
        rounds = k1_rounds([draft_id], [target_id])

    t0_verified = bool(step and step["accepted_draft"] and not args.gate2)
    report = {
        "gate": 3,
        "hidden_source": "none" if args.gate2 else "cli",
        "draft_id": draft_id,
        "target_id": target_id,
        "target_token": args.target_token,
        "probe": probe,
        "k1_step": step,
        "k1_rounds": rounds,
        "t0_identity_verified": t0_verified,
        "note": "rule + live 8K target; dummy-hidden draft is not T=0 identity",
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "gate": 3,
            "draft_id": draft_id,
            "target_id": target_id,
            "target_token": args.target_token,
            "accepted_draft": None if step is None else step["accepted_draft"],
            "t0_identity_verified": t0_verified,
        },
        ensure_ascii=False,
    ))
    if step is None:
        print("MTP_GATE3_PROBE_OK")
        return 0
    print("MTP_GATE3_RULE_OK")
    if t0_verified:
        print("MTP_GATE3_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

