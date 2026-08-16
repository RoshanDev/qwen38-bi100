#!/usr/bin/env python3
"""Measure simple end-to-end generation throughput through the OpenAI API."""

import argparse
import json
import time
import urllib.request


def generate(base_url: str, max_tokens: int) -> tuple[int, float, str]:
    payload = {
        "model": "Qwen3.8-27B",
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n请详细介绍 Go 的并发模型、调度器和 channel。"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    tokens = int(result.get("usage", {}).get("completion_tokens", 0))
    text = result["choices"][0]["text"]
    return tokens, elapsed, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1111")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    warmup_tokens, warmup_elapsed, _ = generate(args.base_url, 32)
    print(
        f"warmup_tokens={warmup_tokens} warmup_elapsed={warmup_elapsed:.3f}s"
    )

    tokens, elapsed, text = generate(args.base_url, args.max_tokens)
    throughput = tokens / elapsed if elapsed else 0.0
    print(f"completion_tokens={tokens}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"end_to_end_throughput={throughput:.3f} tok/s")
    print(f"output_preview={text[:200]!r}")


if __name__ == "__main__":
    main()
