#!/usr/bin/env python3
"""Minimal multi-GPU CoreX runtime smoke test for the remote BI-V100 host."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    reduced = torch.tensor([local_rank + 1.0], device="cuda")
    dist.all_reduce(reduced)

    matrix = torch.randn(
        (1024, 1024),
        device="cuda",
        dtype=torch.float16,
    )
    output = matrix @ matrix
    torch.cuda.synchronize()

    expected = world_size * (world_size + 1) / 2
    print(
        f"rank={local_rank}",
        f"device={torch.cuda.get_device_name(local_rank)}",
        f"all_reduce={reduced.item():.1f}",
        f"matmul_mean={output.float().abs().mean().item():.6f}",
        flush=True,
    )
    if reduced.item() != expected:
        raise RuntimeError(
            f"all-reduce mismatch: expected {expected}, got {reduced.item()}"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
