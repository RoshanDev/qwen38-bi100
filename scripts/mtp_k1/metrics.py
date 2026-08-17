"""Machine-readable MTP K=1 metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RequestMetrics:
    request_id: str
    proposal_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    bonus_token_count: int = 0
    rollback_count: int = 0
    mtp_forward_ms: float = 0.0
    target_verify_ms: float = 0.0
    rollback_ms: float = 0.0
    sampling_ms: float = 0.0
    prefill_ms: float = 0.0
    total_decode_ms: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    ttft_s: float | None = None
    decode_tok_s: float | None = None
    end_to_end_tok_s: float | None = None
    target_kv_tokens: int = 0
    mtp_kv_tokens: int = 0
    sequence_length: int = 0
    finish_reason: str = ""
    token_sha256: str = ""
    gpu_memory_mib: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.proposal_count == 0:
            return 0.0
        return self.accepted_count / self.proposal_count

    @property
    def mean_accepted_tokens(self) -> float:
        if self.proposal_count == 0:
            return 0.0
        return self.accepted_count / self.proposal_count

    def finalize(self, e2e_s: float) -> None:
        if self.output_tokens > 0 and self.total_decode_ms > 0:
            self.decode_tok_s = self.output_tokens / (self.total_decode_ms / 1000.0)
        if self.output_tokens > 0 and e2e_s > 0:
            self.end_to_end_tok_s = self.output_tokens / e2e_s

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["acceptance_rate"] = self.acceptance_rate
        payload["mean_accepted_tokens"] = self.mean_accepted_tokens
        return payload
