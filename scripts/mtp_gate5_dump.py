#!/usr/bin/env python3
"""Dump 128-256 greedy target steps: token, last-layer hidden, real position.

Runs inside the CoreX image on GPU0-3. Does not enable the MTP observer hook.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mtp_k1.trace import assemble_records, classify_dump

DUMP_DIR = Path(os.environ.get("MTP_GATE5_DUMP", "/logs/mtp_gate5_kv"))


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        return 0
    return 0


def install_trace_dump() -> None:
    import torch
    import vllm.model_executor.models.qwen3_5 as qwen35

    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    orig_fwd = qwen35.Qwen3_5ForCausalLM.forward
    orig_logits = qwen35.Qwen3_5ForCausalLM.compute_logits

    def forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None, **kwargs):  # type: ignore[no-untyped-def]
        if _rank() == 0 and positions is not None:
            self._gate5_positions = positions.detach().reshape(-1).to("cpu")
            if input_ids is not None:
                self._gate5_input_ids = input_ids.detach().reshape(-1).to("cpu")
        return orig_fwd(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, **kwargs)

    def compute_logits(self, hidden_states, sampling_metadata):  # type: ignore[no-untyped-def]
        try:
            if _rank() == 0 and hidden_states is not None:
                flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                rows = int(flat.shape[0])
                kind = classify_dump(rows)
                positions = getattr(self, "_gate5_positions", None)
                input_ids = getattr(self, "_gate5_input_ids", None)
                pos_list = [int(v) for v in positions.tolist()] if positions is not None else []
                selected = None
                indices = getattr(sampling_metadata, "selected_token_indices", None)
                if indices is not None and len(indices) > 0:
                    selected = int(indices[-1])
                    hidden = flat[selected]
                    how = "selected_token_indices"
                else:
                    hidden = flat[-1]
                    how = "last_row"
                if kind == "prefill" and selected is not None:
                    position = pos_list[selected] if selected < len(pos_list) else (pos_list[-1] if pos_list else None)
                elif pos_list:
                    position = pos_list[-1]
                else:
                    position = None
                if kind != "skip" and position is not None:
                    idx = len(list(DUMP_DIR.glob("raw_*.pt")))
                    torch.save(
                        {
                            "kind": kind,
                            "how": how,
                            "rows": rows,
                            "shape": list(hidden_states.shape),
                            "selected": selected,
                            "position": int(position),
                            "positions": pos_list if len(pos_list) <= 256 else pos_list[-8:],
                            "input_ids": [int(v) for v in input_ids.tolist()] if input_ids is not None and input_ids.numel() <= 256 else None,
                            "hidden": hidden.detach().to("cpu", dtype=torch.float32),
                        },
                        DUMP_DIR / f"raw_{idx:04d}.pt",
                    )
        except Exception as exc:  # pragma: no cover
            (DUMP_DIR / "dump_error.txt").write_text(str(exc), encoding="utf-8")
        return orig_logits(self, hidden_states, sampling_metadata)

    qwen35.Qwen3_5ForCausalLM.forward = forward
    qwen35.Qwen3_5ForCausalLM.compute_logits = compute_logits


def build_prompt(model_dir: str, user_text: str) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        return str(tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            enable_thinking=False,
            **kwargs,
        ))
    except TypeError:
        return str(tok.apply_chat_template([{"role": "user", "content": user_text}], **kwargs))


def load_raws() -> list[dict[str, Any]]:
    import torch

    files = sorted(DUMP_DIR.glob("raw_*.pt"))
    return [torch.load(path, map_location="cpu") for path in files]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default="用两段中文解释矩阵乘法为什么是 Transformer 的主要计算，不要提纲。")
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["QWEN38_MTP_K1"] = "0"
    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    install_trace_dump()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    prompt = build_prompt(args.model, args.prompt)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=4,
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.80,
        disable_custom_all_reduce=True,
        max_num_seqs=1,
    )
    outs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=int(args.max_tokens), skip_special_tokens=False),
    )
    token_ids = [int(v) for v in outs[0].outputs[0].token_ids]
    raws = load_raws()
    report: dict[str, Any] = {
        "gate": "5-dump",
        "prompt": args.prompt,
        "prompt_len_tokenizer": len(prompt_ids),
        "generated_tokens": token_ids,
        "n_generated": len(token_ids),
        "n_raw_dumps": len(raws),
        "finish_reason": getattr(outs[0].outputs[0], "finish_reason", None),
    }
    try:
        assembled = assemble_records(token_ids, raws)
    except Exception as exc:
        report["error"] = str(exc)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("MTP_GATE5_DUMP_FAIL")
        return 1

    import torch

    usable = [row for row in raws if row.get("kind") in {"prefill", "decode"}]
    hiddens = torch.stack([row["hidden"].float().reshape(-1) for row in usable])
    if int(hiddens.shape[0]) < assembled["n_steps"]:
        report["error"] = f"hidden rows {tuple(hiddens.shape)} < steps {assembled['n_steps']}"
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("MTP_GATE5_DUMP_FAIL")
        return 1
    payload_path = str(Path(args.out).with_suffix(".pt"))
    torch.save(
        {
            "hiddens": hiddens[: assembled["n_steps"]].contiguous(),
            "steps": assembled["steps"],
            "token_ids": token_ids,
            "prompt": args.prompt,
            "prompt_len_tokenizer": len(prompt_ids),
        },
        payload_path,
    )
    report.update(assembled)
    report["payload"] = payload_path
    report["positions_head"] = [row["position"] for row in assembled["steps"][:8]]
    report["positions_tail"] = [row["position"] for row in assembled["steps"][-4:]]
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "gate": "5-dump",
            "n_steps": assembled["n_steps"],
            "n_generated": len(token_ids),
            "positions_head": report["positions_head"],
            "payload": payload_path,
        },
        ensure_ascii=False,
    ))
    print("MTP_GATE5_DUMP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
