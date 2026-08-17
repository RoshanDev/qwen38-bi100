#!/usr/bin/env python3
"""Add independently gated dense GDN fast paths to the installed adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


LOOP1_OLD = """\
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
"""

LOOP1_NEW = """\
    import os as _os
    loop1_native = _os.environ.get("QWEN38_LOOP1_NATIVE", "0") == "1"
    if loop1_native:
        eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        attn = torch.linalg.solve_triangular(
            -attn, eye, upper=False, unitriangular=True)
    else:
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(
            chunk_size, dtype=attn.dtype, device=attn.device)
"""

CHUNK_OLD = """\
    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_upper2, 0)
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        core_out[:, :, i] = attn_inter + attn_i @ v_new
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
            .transpose(-1, -2) @ v_new
        )
"""

CHUNK_NEW = """\
    num_chunks = total_len // chunk_size
    chunk_parallel = _os.environ.get("QWEN38_CHUNK_PARALLEL", "0") == "1"
    if chunk_parallel:
        g_last = g[:, :, :, -1]
        exp_g_last = g_last.exp()
        attn_all = (
            query @ key.transpose(-1, -2) * decay_mask
        ).masked_fill(mask_upper2, 0)
        k_tilde = key * (
            g_last[..., None, None] - g[..., None]
        ).exp()
        q_gated = query * g.exp()[..., None]
        eye = torch.eye(k_dim, device=query.device, dtype=query.dtype)
        transition = (
            exp_g_last[..., None, None] * eye
            - k_tilde.transpose(-1, -2) @ k_cumdecay
        )
        update = k_tilde.transpose(-1, -2) @ value
        output_state = q_gated - attn_all @ k_cumdecay
        output_value = attn_all @ value
        state_starts = torch.empty(
            batch,
            num_heads,
            num_chunks,
            k_dim,
            v_dim,
            device=query.device,
            dtype=query.dtype,
        )
        for i in range(num_chunks):
            state_starts[:, :, i] = last_state
            last_state = transition[:, :, i] @ last_state + update[:, :, i]
        core_out = output_state @ state_starts + output_value
    else:
        for i in range(num_chunks):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn_i = (
                q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
            ).masked_fill_(mask_upper2, 0)
            v_prime = k_cumdecay[:, :, i] @ last_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
            core_out[:, :, i] = attn_inter + attn_i @ v_new
            last_state = (
                last_state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i
                    * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )
"""

RMS_OLD = """\
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        hs = self.weight * hs.to(input_dtype)
        return (hs * F.silu(gate.to(torch.float32))).to(input_dtype)
"""

RMS_NEW = """\
        import os as _os
        if _os.environ.get("QWEN38_RMSNORM_NATIVE", "0") == "1":
            from vllm import _custom_ops as _ops

            out = torch.empty_like(hidden_states)
            _ops.rms_norm(
                out,
                hidden_states.contiguous(),
                self.weight,
                self.variance_epsilon,
            )
            return (
                out.to(torch.float32) * F.silu(gate.to(torch.float32))
            ).to(hidden_states.dtype)
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        hs = self.weight * hs.to(input_dtype)
        return (hs * F.silu(gate.to(torch.float32))).to(input_dtype)
"""


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} block, found {count}")
    return source.replace(old, new, 1)


def patch_source(source: str) -> str:
    source = replace_once(source, LOOP1_OLD, LOOP1_NEW, "loop1")
    source = replace_once(source, CHUNK_OLD, CHUNK_NEW, "chunk recurrence")
    source = replace_once(source, RMS_OLD, RMS_NEW, "RMSNorm")
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("qwen_model_file", type=Path)
    args = parser.parse_args()
    path = args.qwen_model_file
    source = path.read_text(encoding="utf-8")
    patched = patch_source(source)
    temporary = path.with_name(f".{path.name}.native.tmp")
    temporary.write_text(patched, encoding="utf-8")
    temporary.replace(path)
    print(f"patched dense native fast paths: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
