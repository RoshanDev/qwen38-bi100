#!/usr/bin/env python3
"""Patch installed qwen3_5.py to load MTP and expose last hidden + propose()."""
from __future__ import annotations

import argparse
from pathlib import Path


SKIP_OLD = """\
            # Skip vision and MTP branches
            if (name.startswith("model.visual")
                    or name.startswith("mtp.")
                    or name.startswith("model.mtp")):
                continue
"""

SKIP_NEW = """\
            # Skip vision. Load MTP when QWEN38_MTP_K1=1.
            if name.startswith("model.visual"):
                continue
            if name.startswith("mtp.") or name.startswith("model.mtp"):
                import os as _os
                if _os.environ.get("QWEN38_MTP_K1", "0") != "1":
                    continue
                if name.startswith("model.mtp"):
                    name = name[len("model."):]
"""

INIT_OLD = """\
        self.logits_processor = LogitsProcessor(text_cfg.vocab_size)
        self.sampler = Sampler()

        # Lazy initialised in first forward call
        self.mamba_cache: Optional[MambaCacheManager] = None
"""

INIT_NEW = """\
        self.logits_processor = LogitsProcessor(text_cfg.vocab_size)
        self.sampler = Sampler()

        # Lazy initialised in first forward call
        self.mamba_cache: Optional[MambaCacheManager] = None
        self.last_hidden_states = None
        self.last_hidden_meta = {}
"""

LOGITS_OLD = """\
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)
"""

LOGITS_NEW = """\
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # Last-layer hidden after the tokens just computed (final model norm).
        # Row i corresponds to the last computed token of that row.
        self.last_hidden_states = hidden_states
        self.last_hidden_meta = {
            "source": "Qwen3_5Model.forward after final GemmaRMSNorm",
            "norm": "after_final_norm",
            "shape": list(hidden_states.shape),
        }
        return self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)
"""

SAMPLE_OLD = """\
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)
"""

SAMPLE_NEW = """\
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        sampled = self.sampler(logits, sampling_metadata)
        import os as _os
        if _os.environ.get("QWEN38_MTP_K1", "0") == "1":
            from mtp_k1.runtime_hook import after_sample
            after_sample(self, logits, sampled)
        return sampled
"""


def patch_source(source: str) -> str:
    required = (
        (INIT_OLD, INIT_NEW, "mtp init"),
        (LOGITS_OLD, LOGITS_NEW, "logits hook"),
        (SAMPLE_OLD, SAMPLE_NEW, "sample hook"),
    )
    for old, new, label in required:
        if source.count(old) != 1:
            raise RuntimeError(f"expected one {label} block, found {source.count(old)}")
        source = source.replace(old, new)
    source = source.replace(
        "# Text-only (no VL, no MTP).",
        "# Text-only vision skip remains. MTP optional via QWEN38_MTP_K1=1.",
        1,
    )
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    args = parser.parse_args()
    path = Path(args.target)
    original = path.read_text(encoding="utf-8")
    path.write_text(patch_source(original), encoding="utf-8")
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
