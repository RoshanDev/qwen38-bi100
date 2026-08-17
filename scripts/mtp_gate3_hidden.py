#!/usr/bin/env python3
"""Gate 3 live-hidden: dump 64-layer last-token hidden from vLLM, then MTP K=1.

Runs inside the CoreX image on GPU0-3. Do not import this on the host.
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

from mtp_gate2_logits import (  # noqa: E402
    EMBED_KEY,
    LM_HEAD_KEY,
    _load_matrix,
    _text_config,
    mtp_forward,
)
from mtp_gate3_accept import k1_rounds, k1_step  # noqa: E402

DUMP_DIR = Path(os.environ.get("MTP_HIDDEN_DUMP", "/logs/mtp_hidden_dumps"))


def install_hidden_dump() -> None:
    import torch
    import vllm.model_executor.models.qwen3_5 as qwen35

    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    orig = qwen35.Qwen3_5ForCausalLM.compute_logits

    def compute_logits(self, hidden_states, sampling_metadata):  # type: ignore[no-untyped-def]
        try:
            rank = 0
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    rank = int(dist.get_rank())
            except Exception:
                rank = 0
            if rank == 0 and hidden_states is not None:
                flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                chosen = None
                indices = getattr(sampling_metadata, "selected_token_indices", None)
                if indices is not None and len(indices) > 0:
                    chosen = flat[int(indices[-1])]
                    how = "selected_token_indices"
                elif flat.shape[0] <= 256:
                    chosen = flat[-1]
                    how = "last_unpadded"
                else:
                    # Skip the padded 2048-row warmup; next unpadded call is the prompt.
                    how = "skipped_padded"
                    chosen = None
                if chosen is not None:
                    payload = {
                        "hidden": chosen.detach().to("cpu", dtype=torch.float32),
                        "shape": list(hidden_states.shape),
                        "rows": int(flat.shape[0]),
                        "how": how,
                    }
                    idx = len(list(DUMP_DIR.glob("hidden_*.pt")))
                    torch.save(payload, DUMP_DIR / f"hidden_{idx:02d}.pt")
        except Exception as exc:  # pragma: no cover
            (DUMP_DIR / "dump_error.txt").write_text(str(exc), encoding="utf-8")
        return orig(self, hidden_states, sampling_metadata)

    qwen35.Qwen3_5ForCausalLM.compute_logits = compute_logits


def build_prompt(model_dir: str, user_text: str) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return str(tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            enable_thinking=False,
            **kwargs,
        ))
    except TypeError:
        return str(tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            **kwargs,
        ))


def load_dumps() -> list[dict[str, Any]]:
    import torch

    files = sorted(DUMP_DIR.glob("hidden_*.pt"))
    return [torch.load(path, map_location="cpu") for path in files]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default="只回复字母 A，不要其它内容。")
    parser.add_argument("--expected-t0", type=int, default=1596)
    args = parser.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    install_hidden_dump()

    from vllm import LLM, SamplingParams

    from transformers import AutoTokenizer

    prompt = build_prompt(args.model, args.prompt)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_len = len(tok.encode(prompt, add_special_tokens=False))
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
        SamplingParams(temperature=0.0, max_tokens=2, skip_special_tokens=False),
    )
    token_ids = list(outs[0].outputs[0].token_ids)
    dumps = load_dumps()
    if not dumps or len(token_ids) < 1:
        Path(args.out).write_text(
            json.dumps({"gate": 3, "error": "no_hidden_or_tokens", "token_ids": token_ids, "dumps": len(dumps)}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print("MTP_GATE3_HIDDEN_FAIL")
        return 1

    import torch

    text = _text_config(Path(args.model))
    embed = _load_matrix(Path(args.model), EMBED_KEY)
    lm_head = _load_matrix(Path(args.model), LM_HEAD_KEY)
    from mtp_gate1_load import load_mtp_tensors

    mtp = load_mtp_tensors(Path(args.model))
    hidden0 = dumps[0]["hidden"].to(dtype=embed.dtype).view(1, 1, -1)
    t0 = int(token_ids[0])
    target_from_hidden = int(torch.argmax(torch.nn.functional.linear(hidden0.float(), lm_head.float()).reshape(-1)).item())
    logits = mtp_forward(
        torch.tensor([[t0]], dtype=torch.long),
        hidden0,
        embed,
        lm_head,
        mtp,
        rms_eps=float(text["rms_norm_eps"]),
        num_heads=int(text["num_attention_heads"]),
        num_kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        rope_theta=float(text["rope_theta"]),
        partial_rotary_factor=float(text["partial_rotary_factor"]),
        position=max(0, prompt_len),
    )
    draft_id = int(torch.argmax(logits.float().reshape(-1)).item())
    t1 = int(token_ids[1]) if len(token_ids) > 1 else -1
    step = k1_step(draft_id, t1) if t1 >= 0 else None
    rounds = k1_rounds([draft_id], [t1]) if t1 >= 0 else None
    emitted0 = t0
    t0_identity = emitted0 == int(args.expected_t0) and target_from_hidden == t0
    report = {
        "gate": 3,
        "hidden_source": "vllm_qwen3_5_compute_logits",
        "dumps": len(dumps),
        "hidden0_shape": dumps[0].get("shape"),
        "token_ids": token_ids,
        "t0": t0,
        "t1": t1,
        "expected_t0": int(args.expected_t0),
        "target_from_hidden": target_from_hidden,
        "lm_head_matches_t0": target_from_hidden == t0,
        "draft_id": draft_id,
        "k1_step": step,
        "k1_rounds": rounds,
        "t0_identity_verified": t0_identity,
        "prompt_head": prompt[:160],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "gate": 3,
            "t0": t0,
            "t1": t1,
            "draft_id": draft_id,
            "lm_head_matches_t0": target_from_hidden == t0,
            "t0_identity_verified": t0_identity,
            "accepted_draft": None if step is None else step["accepted_draft"],
        },
        ensure_ascii=False,
    ))
    if target_from_hidden != t0:
        print("MTP_GATE3_HIDDEN_MISMATCH")
        return 1
    print("MTP_GATE3_HIDDEN_OK")
    if t0_identity:
        print("MTP_GATE3_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
