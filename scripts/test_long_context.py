#!/usr/bin/env python3
"""Run a real long-context retrieval probe against the OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from typing import Any


FILLER = "背景资料：海风越过蓝色山谷；本段只是填充信息，不含问题所需的识别码。\n"


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def build_messages(repetitions: int, first_code: str, second_code: str) -> list[dict[str, str]]:
    first_at = repetitions // 10
    second_at = repetitions * 9 // 10
    chunks = [
        "请阅读全部资料。最后只输出两处关键记录中的识别码，按出现顺序用英文逗号连接，不要解释。\n"
    ]
    for index in range(repetitions):
        if index == first_at:
            chunks.append(f"关键记录一：第一识别码是 {first_code}。请记住。\n")
        if index == second_at:
            chunks.append(f"关键记录二：第二识别码是 {second_code}。请记住。\n")
        chunks.append(FILLER)
    chunks.append("问题：两处关键记录的识别码依次是什么？只输出两个识别码。")
    return [
        {"role": "system", "content": "You are a precise long-context retrieval assistant."},
        {"role": "user", "content": "".join(chunks)},
    ]


def token_count(
    base_url: str,
    model: str,
    repetitions: int,
    first_code: str,
    second_code: str,
    timeout: int,
) -> int:
    result = post_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "messages": build_messages(repetitions, first_code, second_code),
            "add_generation_prompt": True,
        },
        timeout,
    )
    return int(result["count"])


def fit_repetitions(
    base_url: str,
    model: str,
    target_tokens: int,
    first_code: str,
    second_code: str,
    timeout: int,
) -> tuple[int, int]:
    base_count = token_count(base_url, model, 0, first_code, second_code, timeout)
    sample_repetitions = 100
    sample_count = token_count(
        base_url, model, sample_repetitions, first_code, second_code, timeout
    )
    tokens_per_repeat = (sample_count - base_count) / sample_repetitions
    if tokens_per_repeat <= 0:
        raise RuntimeError("token count did not grow with filler repetitions")

    repetitions = max(0, int((target_tokens - base_count) / tokens_per_repeat))
    count = token_count(base_url, model, repetitions, first_code, second_code, timeout)
    for _ in range(3):
        adjustment = round((target_tokens - count) / tokens_per_repeat)
        if adjustment == 0:
            break
        repetitions = max(0, repetitions + adjustment)
        count = token_count(base_url, model, repetitions, first_code, second_code, timeout)
    return repetitions, count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1112")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--target-input-tokens", type=int, default=16000)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()

    if args.target_input_tokens < 1000:
        parser.error("--target-input-tokens must be at least 1000")

    suffix = str(args.target_input_tokens)
    first_code = f"ALPHA{suffix}X"
    second_code = f"OMEGA{suffix}Z"
    repetitions, input_tokens = fit_repetitions(
        args.base_url,
        args.model,
        args.target_input_tokens,
        first_code,
        second_code,
        args.timeout_seconds,
    )
    messages = build_messages(repetitions, first_code, second_code)
    started = time.perf_counter()
    result = post_json(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": args.max_output_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        args.timeout_seconds,
    )
    elapsed = time.perf_counter() - started
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    passed = first_code in content and second_code in content
    print(
        json.dumps(
            {
                "target_input_tokens": args.target_input_tokens,
                "tokenized_input_tokens": input_tokens,
                "repetitions": repetitions,
                "elapsed_seconds": round(elapsed, 3),
                "response": content,
                "usage": usage,
                "expected_codes": [first_code, second_code],
                "passed": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
