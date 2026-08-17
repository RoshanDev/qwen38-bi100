#!/usr/bin/env python3
"""Gate 1: load the 15 BF16 mtp.* tensors. CPU only, no vLLM, no GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_KEYS = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)


def expected_shapes(hidden: int, intermediate: int, heads: int, kv_heads: int, head_dim: int) -> dict[str, tuple[int, ...]]:
    q_out = heads * head_dim * 2  # attn_output_gate
    kv_out = kv_heads * head_dim
    o_in = heads * head_dim
    return {
        "mtp.fc.weight": (hidden, hidden * 2),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,),
        "mtp.norm.weight": (hidden,),
        "mtp.layers.0.input_layernorm.weight": (hidden,),
        "mtp.layers.0.post_attention_layernorm.weight": (hidden,),
        "mtp.layers.0.mlp.gate_proj.weight": (intermediate, hidden),
        "mtp.layers.0.mlp.up_proj.weight": (intermediate, hidden),
        "mtp.layers.0.mlp.down_proj.weight": (hidden, intermediate),
        "mtp.layers.0.self_attn.q_proj.weight": (q_out, hidden),
        "mtp.layers.0.self_attn.k_proj.weight": (kv_out, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": (kv_out, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": (hidden, o_in),
        "mtp.layers.0.self_attn.q_norm.weight": (head_dim,),
        "mtp.layers.0.self_attn.k_norm.weight": (head_dim,),
    }


def index_mtp_map(model_dir: Path) -> dict[str, str]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    return {key: shard for key, shard in weight_map.items() if key in EXPECTED_KEYS}


def load_mtp_tensors(model_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open

    mapping = index_mtp_map(model_dir)
    missing = [key for key in EXPECTED_KEYS if key not in mapping]
    if missing:
        raise SystemExit(f"missing mtp keys in index: {missing}")

    tensors: dict[str, Any] = {}
    shards = sorted(set(mapping.values()))
    for shard in shards:
        path = model_dir / shard
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key, owner in mapping.items():
                if owner != shard:
                    continue
                tensors[key] = handle.get_tensor(key)
    extra = sorted(set(mapping) - set(EXPECTED_KEYS))
    if extra:
        raise SystemExit(f"unexpected extra mtp keys: {extra}")
    return tensors


def check_shapes(tensors: dict[str, Any], shapes: dict[str, tuple[int, ...]]) -> list[str]:
    problems: list[str] = []
    for key, want in shapes.items():
        got = tuple(tensors[key].shape)
        if got != want:
            problems.append(f"{key}: got {got} want {want}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config") or cfg
    shapes = expected_shapes(
        hidden=int(text["hidden_size"]),
        intermediate=int(text["intermediate_size"]),
        heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
    )
    tensors = load_mtp_tensors(model_dir)
    problems = check_shapes(tensors, shapes)
    report = {
        "gate": 1,
        "mtp_num_hidden_layers": (cfg.get("text_config") or {}).get("mtp_num_hidden_layers"),
        "keys": len(tensors),
        "bytes": {key: int(val.nbytes) for key, val in tensors.items()},
        "shapes": {key: list(val.shape) for key, val in tensors.items()},
        "dtype": {key: str(val.dtype) for key, val in tensors.items()},
        "shape_ok": not problems,
        "problems": problems,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": 1, "keys": len(tensors), "shape_ok": not problems, "problems": problems}, ensure_ascii=False))
    if problems:
        return 1
    print("MTP_GATE1_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
