#!/usr/bin/env python3
"""Gate 2: one MTP draft step -> logits. CPU, no vLLM, does not stop 8K/400K.

Loads embed_tokens + lm_head + 15 mtp.* only. Dummy hidden/token prove the
head wiring. T=0 consistency vs the 64-layer target is Gate 3.
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

from mtp_gate1_load import EXPECTED_KEYS, load_mtp_tensors

EMBED_KEY = "model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"


def rotary_dim(head_dim: int, partial_rotary_factor: float) -> int:
    dim = int(head_dim * partial_rotary_factor)
    if dim <= 0 or dim % 2 != 0 or dim > head_dim:
        raise ValueError(f"invalid rotary_dim={dim} from head_dim={head_dim} factor={partial_rotary_factor}")
    return dim


def _rms_norm(x: Any, weight: Any, eps: float) -> Any:
    import torch

    # GemmaRMSNorm: y * (1 + weight). Standard RMSNorm never accepted t1.
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    return (y * (1.0 + weight.float())).to(dtype=x.dtype)


def _rotate_half(x: Any) -> Any:
    import torch

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Any, position: int, theta: float, rot_dim: int) -> Any:
    import torch

    if rot_dim == 0:
        return x
    device = x.device
    half = rot_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    freqs = position * inv_freq
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype=x.dtype)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype=x.dtype)
    rotated = x[..., :rot_dim] * cos + _rotate_half(x[..., :rot_dim]) * sin
    return torch.cat((rotated, x[..., rot_dim:]), dim=-1)


def _linear(x: Any, weight: Any) -> Any:
    import torch

    return torch.nn.functional.linear(x, weight)


def mtp_forward(
    input_ids: Any,
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
) -> Any:
    """Single-token MTP draft. Shapes: ids [B,S], hidden [B,S,H] -> logits [B,S,V]."""
    import torch

    rot = rotary_dim(head_dim, partial_rotary_factor)
    embeds = embed[input_ids]
    fused_in = torch.cat(
        (
            _rms_norm(embeds, mtp["mtp.pre_fc_norm_embedding.weight"], rms_eps),
            _rms_norm(hidden_states, mtp["mtp.pre_fc_norm_hidden.weight"], rms_eps),
        ),
        dim=-1,
    )
    hidden = _linear(fused_in, mtp["mtp.fc.weight"])
    residual = hidden
    hidden = _rms_norm(hidden, mtp["mtp.layers.0.input_layernorm.weight"], rms_eps)

    batch, seq, _ = hidden.shape
    q_gate = _linear(hidden, mtp["mtp.layers.0.self_attn.q_proj.weight"])
    q, gate = q_gate.chunk(2, dim=-1)
    q = q.view(batch, seq, num_heads, head_dim)
    gate = gate.view(batch, seq, num_heads, head_dim)
    k = _linear(hidden, mtp["mtp.layers.0.self_attn.k_proj.weight"]).view(batch, seq, num_kv_heads, head_dim)
    v = _linear(hidden, mtp["mtp.layers.0.self_attn.v_proj.weight"]).view(batch, seq, num_kv_heads, head_dim)
    q = _rms_norm(q, mtp["mtp.layers.0.self_attn.q_norm.weight"], rms_eps)
    k = _rms_norm(k, mtp["mtp.layers.0.self_attn.k_norm.weight"], rms_eps)
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
    # Match Qwen3_5FullAttention: sigmoid output gate, not SiLU.
    ctx = ctx * torch.sigmoid(gate.float()).to(dtype=ctx.dtype)
    attn_out = _linear(ctx.reshape(batch, seq, num_heads * head_dim), mtp["mtp.layers.0.self_attn.o_proj.weight"])
    hidden = residual + attn_out

    residual = hidden
    hidden = _rms_norm(hidden, mtp["mtp.layers.0.post_attention_layernorm.weight"], rms_eps)
    gate_proj = _linear(hidden, mtp["mtp.layers.0.mlp.gate_proj.weight"])
    up_proj = _linear(hidden, mtp["mtp.layers.0.mlp.up_proj.weight"])
    hidden = _linear(torch.nn.functional.silu(gate_proj) * up_proj, mtp["mtp.layers.0.mlp.down_proj.weight"])
    hidden = residual + hidden
    hidden = _rms_norm(hidden, mtp["mtp.norm.weight"], rms_eps)
    return _linear(hidden, lm_head)


def _load_matrix(model_dir: Path, key: str) -> Any:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = (index.get("weight_map") or {}).get(key)
    if not shard:
        raise SystemExit(f"missing {key} in index")
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _text_config(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = dict(cfg.get("text_config") or cfg)
    rope = text.get("rope_parameters") or {}
    text["rope_theta"] = float(rope.get("rope_theta") or text.get("rope_theta") or 10_000_000)
    text["partial_rotary_factor"] = float(
        rope.get("partial_rotary_factor") or text.get("partial_rotary_factor") or 1.0
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--token-id", type=int, default=248044)
    parser.add_argument("--position", type=int, default=0)
    args = parser.parse_args()
    model_dir = Path(args.model)
    text = _text_config(model_dir)
    hidden_size = int(text["hidden_size"])
    vocab = int(text["vocab_size"])
    token_id = int(args.token_id)
    if not 0 <= token_id < vocab:
        raise SystemExit(f"token-id {token_id} out of vocab {vocab}")

    import torch

    embed = _load_matrix(model_dir, EMBED_KEY)
    lm_head = _load_matrix(model_dir, LM_HEAD_KEY)
    mtp = load_mtp_tensors(model_dir)
    if tuple(embed.shape) != (vocab, hidden_size):
        raise SystemExit(f"embed shape {tuple(embed.shape)} != {(vocab, hidden_size)}")
    if tuple(lm_head.shape) != (vocab, hidden_size):
        raise SystemExit(f"lm_head shape {tuple(lm_head.shape)} != {(vocab, hidden_size)}")

    input_ids = torch.tensor([[token_id]], dtype=torch.long)
    # Zero hidden isolates the embedding path; enough to prove one draft step.
    hidden = torch.zeros((1, 1, hidden_size), dtype=embed.dtype)
    logits = mtp_forward(
        input_ids,
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
    logits_f = logits.float().reshape(-1)
    finite = bool(torch.isfinite(logits_f).all().item())
    topk = torch.topk(logits_f, k=5)
    report = {
        "gate": 2,
        "device": "cpu",
        "backbone_loaded": False,
        "input_token_id": token_id,
        "position": int(args.position),
        "hidden_mode": "zeros",
        "logits_shape": list(logits.shape),
        "finite": finite,
        "logit_min": float(logits_f.min()),
        "logit_max": float(logits_f.max()),
        "logit_mean": float(logits_f.mean()),
        "top1_id": int(topk.indices[0]),
        "top1_logit": float(topk.values[0]),
        "top5": [{"id": int(i), "logit": float(v)} for i, v in zip(topk.indices, topk.values)],
        "rotary_dim": rotary_dim(int(text["head_dim"]), float(text["partial_rotary_factor"])),
        "mtp_keys": len(mtp),
        "note": "head-only dummy hidden; T=0 vs 64-layer target is Gate 3",
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("gate", "finite", "logits_shape", "top1_id", "top1_logit")}, ensure_ascii=False))
    if not finite or list(logits.shape) != [1, 1, vocab]:
        return 1
    print("MTP_GATE2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
