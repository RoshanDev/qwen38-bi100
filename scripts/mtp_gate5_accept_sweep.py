#!/usr/bin/env python3
"""Sweep MTP implementation variants on saved hiddens. CPU only.

Looks for any combo that accepts draft==t1 without stopping 8K.
"""
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
    _apply_rope,
    _linear,
    _load_matrix,
    _text_config,
    rotary_dim,
)
from mtp_gate3_replay import select_hidden_row


def _rms_norm(x: Any, weight: Any, eps: float, gemma: bool) -> Any:
    import torch

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    scale = (1.0 + weight.float()) if gemma else weight.float()
    return (y * scale).to(dtype=x.dtype)


def mtp_logits(
    token_id: int,
    hidden_states: Any,
    embed: Any,
    lm_head: Any,
    mtp: dict[str, Any],
    *,
    rms_eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float,
    position: int,
    gemma_rms: bool,
    concat_hidden_first: bool,
    attn_gate: str,
) -> Any:
    import torch

    rot = rotary_dim(head_dim, partial_rotary_factor)
    embeds = embed[torch.tensor([[int(token_id)]], dtype=torch.long)]
    norm_e = _rms_norm(embeds, mtp["mtp.pre_fc_norm_embedding.weight"], rms_eps, gemma_rms)
    norm_h = _rms_norm(hidden_states, mtp["mtp.pre_fc_norm_hidden.weight"], rms_eps, gemma_rms)
    fused_in = torch.cat((norm_h, norm_e), dim=-1) if concat_hidden_first else torch.cat((norm_e, norm_h), dim=-1)
    hidden = _linear(fused_in, mtp["mtp.fc.weight"])
    residual = hidden
    hidden = _rms_norm(hidden, mtp["mtp.layers.0.input_layernorm.weight"], rms_eps, gemma_rms)

    batch, seq, _ = hidden.shape
    q_gate = _linear(hidden, mtp["mtp.layers.0.self_attn.q_proj.weight"])
    q, gate = q_gate.chunk(2, dim=-1)
    q = q.view(batch, seq, num_heads, head_dim)
    gate = gate.view(batch, seq, num_heads, head_dim)
    k = _linear(hidden, mtp["mtp.layers.0.self_attn.k_proj.weight"]).view(batch, seq, num_kv_heads, head_dim)
    v = _linear(hidden, mtp["mtp.layers.0.self_attn.v_proj.weight"]).view(batch, seq, num_kv_heads, head_dim)
    q = _rms_norm(q, mtp["mtp.layers.0.self_attn.q_norm.weight"], rms_eps, gemma_rms)
    k = _rms_norm(k, mtp["mtp.layers.0.self_attn.k_norm.weight"], rms_eps, gemma_rms)
    q = _apply_rope(q, position, rope_theta, rot)
    k = _apply_rope(k, position, rope_theta, rot)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    repeat = num_heads // num_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)
    attn = torch.softmax(attn, dim=-1).to(dtype=v.dtype)
    ctx = torch.matmul(attn, v).permute(0, 2, 1, 3)
    if attn_gate == "sigmoid":
        ctx = ctx * torch.sigmoid(gate.float()).to(dtype=ctx.dtype)
    elif attn_gate == "silu":
        ctx = ctx * torch.nn.functional.silu(gate)
    else:
        raise ValueError(attn_gate)
    attn_out = _linear(ctx.reshape(batch, seq, num_heads * head_dim), mtp["mtp.layers.0.self_attn.o_proj.weight"])
    hidden = residual + attn_out
    residual = hidden
    hidden = _rms_norm(hidden, mtp["mtp.layers.0.post_attention_layernorm.weight"], rms_eps, gemma_rms)
    gate_proj = _linear(hidden, mtp["mtp.layers.0.mlp.gate_proj.weight"])
    up_proj = _linear(hidden, mtp["mtp.layers.0.mlp.up_proj.weight"])
    hidden = _linear(torch.nn.functional.silu(gate_proj) * up_proj, mtp["mtp.layers.0.mlp.down_proj.weight"])
    hidden = residual + hidden
    hidden = _rms_norm(hidden, mtp["mtp.norm.weight"], rms_eps, gemma_rms)
    return _linear(hidden, lm_head)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dump0", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--t0", type=int, default=32)
    parser.add_argument("--t1", type=int, default=248046)
    args = parser.parse_args()

    import torch

    text = _text_config(Path(args.model))
    embed = _load_matrix(Path(args.model), EMBED_KEY)
    lm_head = _load_matrix(Path(args.model), LM_HEAD_KEY)
    mtp = load_mtp_tensors(Path(args.model))
    payload = torch.load(args.dump0, map_location="cpu")
    hidden_vec, how = select_hidden_row(payload)
    hidden = hidden_vec.to(dtype=embed.dtype).view(1, 1, -1)
    pred0 = int(torch.argmax(torch.nn.functional.linear(hidden.float(), lm_head.float()).reshape(-1)).item())

    hits: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for gemma in (False, True):
        for hidden_first in (False, True):
            for gate in ("sigmoid", "silu"):
                for pos in (19, 20, 21, 22):
                    logits = mtp_logits(
                        int(args.t0),
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
                        position=pos,
                        gemma_rms=gemma,
                        concat_hidden_first=hidden_first,
                        attn_gate=gate,
                    )
                    draft = int(torch.argmax(logits.float().reshape(-1)).item())
                    row = {
                        "gemma_rms": gemma,
                        "concat_hidden_first": hidden_first,
                        "attn_gate": gate,
                        "position": pos,
                        "draft": draft,
                        "accept": draft == int(args.t1),
                    }
                    rows.append(row)
                    if row["accept"]:
                        hits.append(row)

    report = {
        "gate": 5,
        "dump": args.dump0,
        "select": how,
        "pred0": pred0,
        "t0": int(args.t0),
        "t1": int(args.t1),
        "lm_head_matches_t0": pred0 == int(args.t0),
        "tried": len(rows),
        "hits": hits,
        "sample": rows[:4] + rows[-2:],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "gate": 5,
            "tried": len(rows),
            "hits": len(hits),
            "pred0": pred0,
            "first_hit": hits[0] if hits else None,
        },
        ensure_ascii=False,
    ))
    if hits:
        print("MTP_GATE5_ACCEPT_OK")
        return 0
    print("MTP_GATE5_NO_ACCEPT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
