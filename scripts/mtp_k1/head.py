"""Verified MTP K=1 head. Gemma RMSNorm, embed||hidden, sigmoid gate."""
from __future__ import annotations

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

EMBED_KEY = "model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"


def gemma_rms_torch(x, weight, eps: float):
    import torch

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(variance + eps)
    return (y * (1.0 + weight.float())).to(dtype=x.dtype)


def _rotate_half(x):
    import torch

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, position: int, theta: float, rot_dim: int):
    import torch

    if rot_dim == 0:
        return x
    half = rot_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    )
    freqs = position * inv_freq
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype=x.dtype)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype=x.dtype)
    rotated = x[..., :rot_dim] * cos + _rotate_half(x[..., :rot_dim]) * sin
    return torch.cat((rotated, x[..., rot_dim:]), dim=-1)


class MTPHead:
    def __init__(self, tensors: dict[str, Any], embed, lm_head, text: dict[str, Any], census: dict[str, int]):
        self.tensors = tensors
        self.embed = embed
        self.lm_head = lm_head
        self.text = text
        self.census = census

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cpu") -> "MTPHead":
        from safetensors import safe_open
        import torch

        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        text = dict(cfg.get("text_config") or cfg)
        rope = text.get("rope_parameters") or {}
        text["rope_theta"] = float(rope.get("rope_theta") or text.get("rope_theta") or 10_000_000)
        text["partial_rotary_factor"] = float(
            rope.get("partial_rotary_factor") or text.get("partial_rotary_factor") or 0.25
        )
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") or {}
        wanted = set(EXPECTED_KEYS) | {EMBED_KEY, LM_HEAD_KEY}
        found = {k: weight_map[k] for k in wanted if k in weight_map}
        missing = sorted(wanted - set(found))
        unexpected = []
        tensors: dict[str, Any] = {}
        for shard in sorted(set(found.values())):
            with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
                for key, owner in found.items():
                    if owner != shard:
                        continue
                    tensors[key] = handle.get_tensor(key).to(device)
        skipped = 0
        census = {
            "loaded": len(EXPECTED_KEYS) - len([k for k in EXPECTED_KEYS if k not in tensors]),
            "missing": len([k for k in EXPECTED_KEYS if k not in tensors]),
            "unexpected": len(unexpected),
            "skipped": skipped,
        }
        if missing:
            raise RuntimeError(f"MTP load missing keys: {missing}")
        embed = tensors.pop(EMBED_KEY)
        lm_head = tensors.pop(LM_HEAD_KEY)
        return cls(tensors, embed, lm_head, text, census)

    def propose(self, token_id: int, hidden, position: int):
        """hidden: [1,1,H] last-layer after final norm, the row that produced the last token."""
        import torch

        mtp = self.tensors
        text = self.text
        hidden_states = hidden
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.view(1, 1, -1)
        elif hidden_states.dim() == 2:
            hidden_states = hidden_states[-1:].unsqueeze(0)
        device = hidden_states.device
        dtype = hidden_states.dtype
        ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        embeds = self.embed.to(device=device, dtype=dtype)[ids]
        def w(name: str):
            return mtp[name].to(device=device, dtype=dtype)
        rms_eps = float(text["rms_norm_eps"])
        num_heads = int(text["num_attention_heads"])
        num_kv_heads = int(text["num_key_value_heads"])
        head_dim = int(text["head_dim"])
        rot = int(head_dim * float(text["partial_rotary_factor"]))
        fused_in = torch.cat(
            (
                gemma_rms_torch(embeds, w("mtp.pre_fc_norm_embedding.weight"), rms_eps),
                gemma_rms_torch(hidden_states, w("mtp.pre_fc_norm_hidden.weight"), rms_eps),
            ),
            dim=-1,
        )
        def linear(x, weight):
            # CPU Half/BF16 addmm is missing; compute in fp32.
            return torch.nn.functional.linear(x.float(), weight.float()).to(dtype=x.dtype)

        hidden = linear(fused_in, w("mtp.fc.weight"))
        residual = hidden
        hidden = gemma_rms_torch(hidden, w("mtp.layers.0.input_layernorm.weight"), rms_eps)
        batch, seq, _ = hidden.shape
        q_gate = linear(hidden, w("mtp.layers.0.self_attn.q_proj.weight"))
        q, gate = q_gate.chunk(2, dim=-1)
        q = q.view(batch, seq, num_heads, head_dim)
        gate = gate.view(batch, seq, num_heads, head_dim)
        k = linear(hidden, w("mtp.layers.0.self_attn.k_proj.weight")).view(
            batch, seq, num_kv_heads, head_dim
        )
        v = linear(hidden, w("mtp.layers.0.self_attn.v_proj.weight")).view(
            batch, seq, num_kv_heads, head_dim
        )
        q = gemma_rms_torch(q, w("mtp.layers.0.self_attn.q_norm.weight"), rms_eps)
        k = gemma_rms_torch(k, w("mtp.layers.0.self_attn.k_norm.weight"), rms_eps)
        q = apply_rope(q, int(position), float(text["rope_theta"]), rot)
        k = apply_rope(k, int(position), float(text["rope_theta"]), rot)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v.float()).permute(0, 2, 1, 3)
        ctx = (ctx * torch.sigmoid(gate.float())).to(dtype=hidden.dtype)
        attn_out = linear(
            ctx.reshape(batch, seq, num_heads * head_dim),
            w("mtp.layers.0.self_attn.o_proj.weight"),
        )
        hidden = residual + attn_out
        residual = hidden
        hidden = gemma_rms_torch(hidden, w("mtp.layers.0.post_attention_layernorm.weight"), rms_eps)
        gate_proj = linear(hidden, w("mtp.layers.0.mlp.gate_proj.weight"))
        up_proj = linear(hidden, w("mtp.layers.0.mlp.up_proj.weight"))
        hidden = linear(
            torch.nn.functional.silu(gate_proj) * up_proj,
            w("mtp.layers.0.mlp.down_proj.weight"),
        )
        hidden = residual + hidden
        hidden = gemma_rms_torch(hidden, w("mtp.norm.weight"), rms_eps)
        logits = linear(hidden, self.lm_head.to(device=hidden.device, dtype=hidden.dtype))
        return int(torch.argmax(logits.float().reshape(-1)).item()), logits
