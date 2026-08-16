#!/usr/bin/env python3
"""Exercise health, model listing, completion, and chat endpoints."""

import argparse
import json
import time
import urllib.error
import urllib.request


def request_json(url: str, payload: dict | None = None, timeout: int = 3600) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1111")
    parser.add_argument("--wait-seconds", type=int, default=3600)
    args = parser.parse_args()

    deadline = time.monotonic() + args.wait_seconds
    health_url = f"{args.base_url}/health"
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"server did not become healthy: {health_url}")
        time.sleep(5)

    models = request_json(f"{args.base_url}/v1/models")
    model_ids = [item["id"] for item in models.get("data", [])]
    if "Qwen3.8-27B" not in model_ids:
        raise RuntimeError(f"served model is missing: {model_ids}")
    print(f"health=ok models={model_ids}")

    completion = request_json(
        f"{args.base_url}/v1/completions",
        {
            "model": "Qwen3.8-27B",
            "prompt": (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n用三句话解释 Go 的 goroutine 与操作系统线程的区别。"
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
            "temperature": 0.0,
            "max_tokens": 128,
        },
    )
    completion_text = completion["choices"][0]["text"]
    if not completion_text.strip():
        raise RuntimeError("completion endpoint returned empty text")
    if "goroutine" not in completion_text.lower() and "线程" not in completion_text:
        raise RuntimeError(f"completion answer is not coherent: {completion_text!r}")
    print("completion_usage", completion.get("usage"))
    print("completion_text", completion_text)

    chat = request_json(
        f"{args.base_url}/v1/chat/completions",
        {
            "model": "Qwen3.8-27B",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "只回答 6：2 加 4 等于多少？",
                },
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    chat_text = chat["choices"][0]["message"]["content"]
    if chat_text.strip() != "6":
        raise RuntimeError(f"chat endpoint returned an unexpected answer: {chat_text!r}")
    print("chat_usage", chat.get("usage"))
    print("chat_text", chat_text)
    print("smoke=ok")


if __name__ == "__main__":
    main()
