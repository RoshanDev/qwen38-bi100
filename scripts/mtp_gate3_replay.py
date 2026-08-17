#!/usr/bin/env python3
"""Replay Gate 3 from a saved last-token hidden. CPU only, no vLLM, no 8K stop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from mtp_gate3_accept import k1_rounds, k1_step


def select_hidden_row(payload: dict) -> tuple[object, str]:
    hidden = payload["hidden"]
    shape = list(payload.get("shape") or list(hidden.shape))
    rows = int(payload.get("rows") or (shape[0] if shape else 1))
    # Prefer unpadded prefill (prompt-length rows) over the 2048 padded probe.
    if hidden.dim() == 1:
        return hidden, "saved_vector"
    if rows <= 256:
        return hidden.reshape(-1, hidden.shape[-1])[-1], "last_unpadded"
    return hidden.reshape(-1, hidden.shape[-1])[-1], "last_padded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--t0", type=int, required=True)
    parser.add_argument("--t1", type=int, required=True)
    parser.add_argument("--position", type=int, default=20)
    args = parser.parse_args()

    import torch

    payload = torch.load(args.dump, map_location="cpu")
    hidden_vec, how = select_hidden_row(payload)
    text = _text_config(Path(args.model))
    embed = _load_matrix(Path(args.model), EMBED_KEY)
    lm_head = _load_matrix(Path(args.model), LM_HEAD_KEY)
    mtp = load_mtp_tensors(Path(args.model))
    hidden = hidden_vec.to(dtype=embed.dtype).view(1, 1, -1)
    target_from_hidden = int(
        torch.argmax(torch.nn.functional.linear(hidden.float(), lm_head.float()).reshape(-1)).item()
    )
    logits = mtp_forward(
        torch.tensor([[int(args.t0)]], dtype=torch.long),
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
        position=int(args.position),
    )
    draft_id = int(torch.argmax(logits.float().reshape(-1)).item())
    step = k1_step(draft_id, int(args.t1))
    rounds = k1_rounds([draft_id], [int(args.t1)])
    match = target_from_hidden == int(args.t0)
    report = {
        "gate": 3,
        "mode": "replay",
        "dump": str(args.dump),
        "select": how,
        "dump_shape": list(payload.get("shape") or []),
        "dump_rows": payload.get("rows"),
        "t0": int(args.t0),
        "t1": int(args.t1),
        "target_from_hidden": target_from_hidden,
        "lm_head_matches_t0": match,
        "draft_id": draft_id,
        "k1_step": step,
        "k1_rounds": rounds,
        "t0_identity_verified": match,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "gate": 3,
            "select": how,
            "target_from_hidden": target_from_hidden,
            "t0": int(args.t0),
            "draft_id": draft_id,
            "lm_head_matches_t0": match,
            "accepted_draft": step["accepted_draft"],
        },
        ensure_ascii=False,
    ))
    if not match:
        print("MTP_GATE3_REPLAY_MISMATCH")
        return 1
    print("MTP_GATE3_HIDDEN_OK")
    print("MTP_GATE3_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
