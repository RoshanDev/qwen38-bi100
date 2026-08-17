#!/usr/bin/env python3
"""Gate 4: two-round K=1 state from saved hiddens. CPU only, no 8K stop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mtp_gate1_load import load_mtp_tensors
from mtp_gate2_logits import (
    EMBED_KEY,
    LM_HEAD_KEY,
    _load_matrix,
    _text_config,
    mtp_forward,
)
from mtp_gate3_accept import k1_pipeline
from mtp_gate3_replay import select_hidden_row


def _argmax_lm(hidden: Any, lm_head: Any) -> int:
    import torch

    return int(torch.argmax(torch.nn.functional.linear(hidden.float(), lm_head.float()).reshape(-1)).item())


def _draft(token_id: int, hidden: Any, embed: Any, lm_head: Any, mtp: dict, text: dict[str, Any], position: int) -> int:
    import torch

    logits = mtp_forward(
        torch.tensor([[int(token_id)]], dtype=torch.long),
        hidden,
        embed,
        lm_head,
        mtp,
        rms_eps=float(text["rms_norm_eps"]),
        num_heads=int(text["num_attention_heads"]),
        num_kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        rope_theta=float(text["rope_theta"]),
        partial_rotary_factor=float(text["partial_rotary_factor"]),
        position=int(position),
    )
    return int(torch.argmax(logits.float().reshape(-1)).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dump0", required=True)
    parser.add_argument("--dump1", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--t0", type=int, default=32)
    parser.add_argument("--t1", type=int, default=248046)
    args = parser.parse_args()

    import torch

    text = _text_config(Path(args.model))
    embed = _load_matrix(Path(args.model), EMBED_KEY)
    lm_head = _load_matrix(Path(args.model), LM_HEAD_KEY)
    mtp = load_mtp_tensors(Path(args.model))

    p0 = torch.load(args.dump0, map_location="cpu")
    p1 = torch.load(args.dump1, map_location="cpu")
    h0_vec, how0 = select_hidden_row(p0)
    h1_vec, how1 = select_hidden_row(p1)
    h0 = h0_vec.to(dtype=embed.dtype).view(1, 1, -1)
    h1 = h1_vec.to(dtype=embed.dtype).view(1, 1, -1)
    pred0 = _argmax_lm(h0, lm_head)
    pred1 = _argmax_lm(h1, lm_head)
    draft1 = _draft(int(args.t0), h0, embed, lm_head, mtp, text, 20)
    draft2 = _draft(int(args.t1), h1, embed, lm_head, mtp, text, 21)
    pipe = k1_pipeline(int(args.t0), [draft1, draft2], [int(args.t1)])
    report = {
        "gate": 4,
        "mode": "two_hidden_rounds",
        "dump0": {"path": args.dump0, "how": how0, "shape": p0.get("shape"), "pred": pred0},
        "dump1": {"path": args.dump1, "how": how1, "shape": p1.get("shape"), "pred": pred1},
        "t0": int(args.t0),
        "t1": int(args.t1),
        "lm_head_matches_t0": pred0 == int(args.t0),
        "lm_head_matches_t1": pred1 == int(args.t1),
        "draft1": draft1,
        "draft2": draft2,
        "pipeline": pipe,
        "rounds_recorded": len(pipe["rounds"]),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "gate": 4,
            "pred0": pred0,
            "pred1": pred1,
            "draft1": draft1,
            "draft2": draft2,
            "emitted": pipe["emitted_ids"],
            "t0_identity": pipe["t0_identity"],
            "matches_no_mtp_prefix": pipe["matches_no_mtp_prefix"],
            "rounds": len(pipe["rounds"]),
        },
        ensure_ascii=False,
    ))
    if pred0 != int(args.t0) or not pipe["t0_identity"] or not pipe["matches_no_mtp_prefix"]:
        print("MTP_GATE4_MISMATCH")
        return 1
    print("MTP_GATE4_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
