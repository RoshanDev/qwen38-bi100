"""In-worker MTP K=1 hook. Called from patched Qwen3_5ForCausalLM.sample."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_HEAD = None
_PENDING: dict[str, Any] = {"draft": None, "position": 0}
_CUMUL = {"proposal_count": 0, "accepted_count": 0, "rejected_count": 0}
_LOG = Path(os.environ.get("MTP_LOG_DIR", "/logs/mtp-k1-exp"))


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _head():
    global _HEAD
    if _HEAD is None:
        from mtp_k1.head import MTPHead

        # Only rank 0 proposes. Avoid 4-way copies of embed/lm_head.
        _HEAD = MTPHead.load(os.environ.get("MTP_MODEL", "/model"), device="cpu")
        _LOG.mkdir(parents=True, exist_ok=True)
        (_LOG / "hook_census.json").write_text(
            json.dumps(_HEAD.census, indent=2) + "\n", encoding="utf-8"
        )
        print("MTP_HOOK_CENSUS " + json.dumps(_HEAD.census), flush=True)
        if any(_HEAD.census[k] for k in ("missing", "unexpected", "skipped")):
            raise RuntimeError(f"unclean MTP census: {_HEAD.census}")
    return _HEAD


def _sampled_token(sampler_output: Any) -> int | None:
    if sampler_output is None:
        return None
    if getattr(sampler_output, "sampled_token_ids", None) is not None:
        ids = sampler_output.sampled_token_ids
        return int(ids.view(-1)[0].item())
    outputs = getattr(sampler_output, "outputs", None)
    if not outputs:
        return None
    first = outputs[0]
    samples = getattr(first, "samples", None)
    if samples:
        return int(samples[0].output_token)
    seqs = getattr(first, "samples", None)
    if hasattr(first, "samples") and first.samples:
        return int(first.samples[0].output_token)
    return None


def after_sample(model: Any, logits: Any, sampler_output: Any) -> None:
    if os.environ.get("QWEN38_MTP_K1", "0") != "1":
        return
    if _rank() != 0:
        return
    token = _sampled_token(sampler_output)
    hidden = getattr(model, "last_hidden_states", None)
    if token is None or hidden is None:
        raise RuntimeError("MTP hook missing token or last_hidden_states")
    # Prefill/profile dumps many rows. Do not run MTP there (also avoids
    # vLLM profile_run crashing before the API is up).
    rows = 1
    if hasattr(hidden, "shape") and len(hidden.shape) >= 2:
        rows = int(hidden.reshape(-1, hidden.shape[-1]).shape[0])
    if rows > 8:
        _PENDING["draft"] = None
        _PENDING["position"] = rows - 1
        return
    head = _head()
    draft_prev = _PENDING.get("draft")
    if draft_prev is not None:
        _CUMUL["proposal_count"] += 1
        if int(draft_prev) == int(token):
            _CUMUL["accepted_count"] += 1
            event = "accept"
        else:
            _CUMUL["rejected_count"] += 1
            event = "reject"
        if _rank() == 0:
            rec = {
                "event": event,
                "draft": int(draft_prev),
                "target": int(token),
                **_CUMUL,
            }
            (_LOG / "hook_events.jsonl").open("a", encoding="utf-8").write(
                json.dumps(rec) + "\n"
            )
    row = hidden
    if hasattr(row, "reshape"):
        row = row.reshape(-1, row.shape[-1])[-1].detach().to("cpu")
    started = time.perf_counter()
    draft, _ = head.propose(int(token), row, position=_PENDING["position"])
    _PENDING["draft"] = draft
    _PENDING["position"] = int(_PENDING["position"]) + 1
    if _rank() == 0:
        rec = {
            "event": "propose",
            "token": int(token),
            "draft": int(draft),
            "mtp_forward_ms": (time.perf_counter() - started) * 1000.0,
            **_CUMUL,
        }
        (_LOG / "hook_events.jsonl").open("a", encoding="utf-8").write(
            json.dumps(rec) + "\n"
        )
