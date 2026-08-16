#!/usr/bin/env python3
"""Prepare a Qwen3.8 checkpoint directory for the text-only vLLM adapter."""

import argparse
import json
import shutil
from pathlib import Path


TARGET_ARCHITECTURE = "Qwen3_5ForCausalLM"
OFFICIAL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    config_path = args.model_dir / "config.json"
    backup_path = args.model_dir / "config.json.qwen38-official"
    original_text = config_path.read_text(encoding="utf-8")
    config = json.loads(original_text)

    architecture = config.get("architectures")
    allowed = ([OFFICIAL_ARCHITECTURE], [TARGET_ARCHITECTURE])
    if architecture not in allowed:
        raise RuntimeError(f"unexpected architectures value: {architecture!r}")
    if config.get("model_type") != "qwen3_5":
        raise RuntimeError(f"unexpected model_type: {config.get('model_type')!r}")

    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)

    if architecture == [OFFICIAL_ARCHITECTURE]:
        if original_text.count(OFFICIAL_ARCHITECTURE) != 1:
            raise RuntimeError("official architecture is not unique in config.json")
        prepared_text = original_text.replace(
            OFFICIAL_ARCHITECTURE,
            TARGET_ARCHITECTURE,
            1,
        )
        config_path.write_text(prepared_text, encoding="utf-8")
    print(f"prepared {config_path}")
    print(f"official backup {backup_path}")


if __name__ == "__main__":
    main()
