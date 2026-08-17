#!/usr/bin/env python3
"""Create a space-efficient Qwen3.8 YaRN model overlay using hard links."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def build_layers_block_type(text_config: dict) -> list[str]:
    """Translate Qwen's hybrid layer layout for vLLM 0.6.3 KV sizing."""
    layer_types = text_config.get("layer_types")
    num_layers = int(text_config.get("num_hidden_layers", 0))
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise RuntimeError(
            "text_config.layer_types length does not match "
            f"num_hidden_layers: {len(layer_types or [])} != {num_layers}"
        )

    mapping = {
        "full_attention": "attention",
        "linear_attention": "linear_attention",
    }
    unknown = sorted(set(layer_types) - set(mapping))
    if unknown:
        raise RuntimeError(f"unsupported Qwen layer types: {unknown}")
    translated = [mapping[layer_type] for layer_type in layer_types]
    attention_layers = translated.count("attention")
    if attention_layers == 0 or attention_layers == num_layers:
        raise RuntimeError(
            "expected a hybrid attention layout, got "
            f"{attention_layers}/{num_layers} attention layers"
        )
    return translated


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--target-context", type=int, default=1_000_000)
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.absolute()
    source_config_path = source / "config.json"
    if not source_config_path.is_file():
        parser.error(f"source config is missing: {source_config_path}")
    if not (source / "model.safetensors.index.json").is_file():
        parser.error(f"source model index is missing: {source}")
    if target.exists():
        parser.error(f"target already exists: {target}")
    if args.factor <= 1:
        parser.error("--factor must be greater than 1")
    if args.target_context <= 262_144:
        parser.error("--target-context must exceed the native 262144 context")

    source_hash_before = sha256(source_config_path)
    with source_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise RuntimeError("config.json has no text_config object")
    native_context = int(text_config.get("max_position_embeddings", 0))
    if native_context != 262_144:
        raise RuntimeError(
            f"unexpected native max_position_embeddings: {native_context}"
        )

    rope_parameters = dict(text_config.get("rope_parameters") or {})
    rope_parameters.update(
        {
            "rope_type": "yarn",
            "factor": args.factor,
            "original_max_position_embeddings": native_context,
        }
    )
    text_config["rope_parameters"] = rope_parameters

    # vLLM 0.6.3 reads this compatibility field from the top-level config.
    # Without it, all 64 Qwen hybrid layers are treated as full-attention KV
    # layers even though only 16 layers consume a KV cache in qwen3_5.py.
    layers_block_type = build_layers_block_type(text_config)
    config["layers_block_type"] = layers_block_type
    attention_layers = layers_block_type.count("attention")

    try:
        shutil.copytree(source, target, copy_function=os.link, symlinks=True)
        target_config_path = target / "config.json"
        target_config_path.unlink()
        temporary_config = target / ".config.json.tmp"
        with temporary_config.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_config, target_config_path)

        manifest = {
            "source_model_dir": str(source),
            "source_config_sha256": source_hash_before,
            "native_context": native_context,
            "target_context": args.target_context,
            "rope_type": "yarn",
            "factor": args.factor,
            "vllm_layers_block_type": "top-level compatibility field",
            "num_hidden_layers": len(layers_block_type),
            "num_attention_layers": attention_layers,
            "storage": "hard-linked overlay; config.json is independent",
        }
        with (target / "qwen38-yarn-overlay.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        raise

    source_hash_after = sha256(source_config_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source config changed while creating the overlay")

    print(
        json.dumps(
            {
                "target": str(target),
                "source_config_sha256": source_hash_after,
                "target_config_sha256": sha256(target / "config.json"),
                "factor": args.factor,
                "target_context": args.target_context,
                "num_hidden_layers": len(layers_block_type),
                "num_attention_layers": attention_layers,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
