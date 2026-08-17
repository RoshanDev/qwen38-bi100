#!/usr/bin/env python3
"""HTTP eval for isolated MTP K=1 server. Writes JSON under --out-dir."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPTS = {
    "zh_explain": "用两段中文解释矩阵乘法为什么是 Transformer 的主要计算，不要提纲。",
    "go_code": "写一个 Go 函数 Reverse(s string) string，只要代码。",
    "json_struct": "只输出 JSON：{\"ok\":true,\"n\":3}",
    "free_text": "随便写一段很难预测的梦境描写。",
    "code_cont": "继续写：func main() {\n    fmt.Println(",
}


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 900) -> dict[str, Any]:
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(base: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    started = time.perf_counter()
    data = http_json(
        base.rstrip("/") + "/v1/chat/completions",
        {
            "model": "Qwen3.8-27B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    data["_wall_s"] = time.perf_counter() - started
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content")) or ""
    data["_text"] = text
    data["_sha"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    data["_finish"] = choice.get("finish_reason")
    return data


def wait_health(base: str, timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise SystemExit(f"health timeout {base}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--prompts", nargs="+", default=["zh_explain", "go_code"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wait_health(args.base_url)
    report: dict[str, Any] = {"label": args.label, "base_url": args.base_url, "groups": []}
    for name in args.prompts:
        prompt = PROMPTS[name]
        for max_tokens in args.max_tokens:
            group: dict[str, Any] = {"prompt": name, "max_tokens": max_tokens, "warmup": [], "formal": []}
            for i in range(args.warmup):
                row = chat(args.base_url, prompt, max_tokens)
                group["warmup"].append({"i": i, "sha": row["_sha"], "finish": row["_finish"], "wall_s": row["_wall_s"], "usage": row.get("usage")})
            for i in range(args.runs):
                row = chat(args.base_url, prompt, max_tokens)
                group["formal"].append(
                    {
                        "i": i,
                        "sha": row["_sha"],
                        "finish": row["_finish"],
                        "wall_s": row["_wall_s"],
                        "usage": row.get("usage"),
                        "text_head": row["_text"][:80],
                    }
                )
                print(json.dumps({"label": args.label, "prompt": name, "max_tokens": max_tokens, "i": i, "sha": row["_sha"], "wall_s": row["_wall_s"]}, ensure_ascii=False), flush=True)
            shas = {r["sha"] for r in group["formal"]}
            group["outputs_identical"] = len(shas) == 1
            group["sha"] = sorted(shas)
            report["groups"].append(group)
    (out_dir / f"{args.label}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE", out_dir / f"{args.label}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
