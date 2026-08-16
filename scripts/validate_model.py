#!/usr/bin/env python3
"""Validate the pinned Qwen3.8-27B checkpoint without loading tensor data."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open


EXPECTED_SHARDS = 18
EXPECTED_FILE_BYTES = 55_563_006_776
EXPECTED_TENSOR_BYTES = 55_562_855_904
EXPECTED_KEYS = 1_199
EXPECTED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    metadata_files = sorted(
        (model_dir / ".cache" / "huggingface" / "download").glob("*.metadata")
    )
    revisions = {
        path.read_text(encoding="utf-8").splitlines()[0]
        for path in metadata_files
    }
    if revisions != {EXPECTED_REVISION}:
        raise RuntimeError(f"unexpected downloaded revisions: {sorted(revisions)}")

    weight_map = index["weight_map"]
    shards = sorted({model_dir / name for name in weight_map.values()})
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(f"expected {EXPECTED_SHARDS} shards, found {len(shards)}")
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing shards: {missing}")

    file_bytes = sum(path.stat().st_size for path in shards)
    tensor_bytes = index.get("metadata", {}).get("total_size")
    if file_bytes != EXPECTED_FILE_BYTES:
        raise RuntimeError(
            f"unexpected shard bytes: expected {EXPECTED_FILE_BYTES}, got {file_bytes}"
        )
    if tensor_bytes != EXPECTED_TENSOR_BYTES:
        raise RuntimeError(
            f"unexpected tensor bytes: expected {EXPECTED_TENSOR_BYTES}, got {tensor_bytes}"
        )
    if len(weight_map) != EXPECTED_KEYS:
        raise RuntimeError(
            f"unexpected index key count: expected {EXPECTED_KEYS}, got {len(weight_map)}"
        )

    actual_keys: set[str] = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            actual_keys.update(handle.keys())
    indexed_keys = set(weight_map)
    if actual_keys != indexed_keys:
        raise RuntimeError(
            "safetensors/index mismatch: "
            f"missing={sorted(indexed_keys - actual_keys)[:10]} "
            f"extra={sorted(actual_keys - indexed_keys)[:10]}"
        )

    architectures = config.get("architectures")
    if architectures not in (
        ["Qwen3_5ForConditionalGeneration"],
        ["Qwen3_5ForCausalLM"],
    ):
        raise RuntimeError(f"unexpected architectures: {architectures!r}")

    print(f"model_dir={model_dir}")
    print(f"shards={len(shards)} file_bytes={file_bytes}")
    print(f"tensor_bytes={tensor_bytes} keys={len(actual_keys)}")
    print(f"revision={EXPECTED_REVISION} metadata_files={len(metadata_files)}")
    print(f"architectures={architectures}")
    print("validation=ok")


if __name__ == "__main__":
    main()
