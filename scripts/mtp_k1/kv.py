"""One-layer MTP full-attention KV cache.

Stores RoPE'd K/V before GQA repeat. Query attends to all cached positions.
Passing cache=None keeps the old 1x1 observer path.
"""
from __future__ import annotations

from typing import Any


class MTPKVCache:
    """Append-only decode cache. k/v shapes: [B, H_kv, T, D]."""

    def __init__(self) -> None:
        self.k: Any = None
        self.v: Any = None

    def __len__(self) -> int:
        if self.k is None:
            return 0
        return int(self.k.shape[2])

    def append(self, k: Any, v: Any) -> tuple[Any, Any]:
        if k.shape != v.shape:
            raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        if k.dim() != 4:
            raise ValueError(f"expected k/v [B,H,T,D], got {tuple(k.shape)}")
        piece_k = k.detach().contiguous()
        piece_v = v.detach().contiguous()
        if self.k is None:
            self.k = piece_k
            self.v = piece_v
        else:
            if self.k.shape[0] != piece_k.shape[0] or self.k.shape[1] != piece_k.shape[1] or self.k.shape[3] != piece_k.shape[3]:
                raise ValueError(
                    f"cache shape {tuple(self.k.shape)} incompatible with {tuple(piece_k.shape)}"
                )
            import torch

            self.k = torch.cat((self.k, piece_k), dim=2)
            self.v = torch.cat((self.v, piece_v), dim=2)
        return self.k, self.v


def attend_decode(q: Any, k: Any, v: Any, head_dim: int) -> Any:
    """q: [B,H,1,D]; k/v: [B,H,S,D] after GQA repeat. Causal by construction."""
    import torch

    attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (float(head_dim) ** -0.5)
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v.float())
