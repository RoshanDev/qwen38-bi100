# MTP K=1 isolated experiment

Branch: `exp/mtp-k1-vllm`  
Image: `qwen38-bi100:corex3.2.3-mtp-k1-exp`  
Container: `qwen38-bi100-mtp-k1-exp`  
Port: `1113`

## RMSNorm

```
normalized = x / sqrt(mean(x^2) + eps)
output = normalized * (1.0 + weight)   # Gemma, required
```

Do not use `normalized * weight`.

## Status (2026-08-17 lab run)

Image `qwen38-bi100:corex3.2.3-mtp-k1-exp` (`e6e97072`) built. Experiment used GPU0–3 / `:1113`. 8K was stopped, not removed, then restored.

| Item | Result |
|------|--------|
| MTP=0 64-token `zh_explain` | 8.5–8.8s, SHA `cc419445b680...` |
| MTP=1 64-token `zh_explain` | 294.0s, SHA `cc419445b680...` **identical** |
| Hook census | loaded=15 missing=0 unexpected=0 skipped=0 |
| Serving hook | propose=64, accept=9, reject=53, rate **14.5%** |
| 8K after restore | `:1111=200`, chat `OK` |
| 400K throughout | `:1112=200`, chat `OK` |

K=1 in this container is **propose + compare against the next target greedy token**. It does **not** commit a draft, skip a target decode, or roll back. Output SHA matching MTP=0 is expected: the hook never changes the target path.

The 14.5% figure is **draft top-1 hit rate**, not a speculative accepted-token rate. The 294s figure is **CPU observer cost** (stateless 1×1 attention, FP32 Linear, full 248K `lm_head` on rank 0). It is not GPU MTP performance.

This only proves that **this observer hook does not accelerate**. It does **not** prove that Qwen3.8 native MTP has no value on BI-V100.

Known observer gaps:

- MTP Full Attention has no draft KV cache; softmax is 1×1 every step.
- Position is a process-global counter after `rows-1`, not vLLM `positions`.
- No batched target verification.

Do not continue observer 256/1024, 10-turn, or cancel-recovery tests. Do not implement fused 2-token verify until Gate 5 (recorded position + one-layer MTP KV) says the hit rate is worth it.

## Gate 5 offline replay (2026-08-18)

Deterministic target dump: `zh_explain`, T=0, thinking off, 160 tokens, `finish_reason=length`. Prompt tokenizer length 28. Recorded positions 27…185, strictly +1 each step. `lm_head(hidden)==sampled_token` on **159/159** steps.

| Pass | top-1 | top-5 | mean rank | median rank |
|------|------:|------:|----------:|------------:|
| Official: MTP KV + recorded position | **15.7%** (25/159) | 44.7% | 87.0 | 9 |
| Control: 1×1, no KV | 16.4% (26/159) | 40.3% | 106.3 | 9 |
| Diagnostic position −2/−1/+1/+2 (first 32, cache on) | 12.5% / 15.6% / 12.5% / 12.5% | ~41% | worse | — |

Decision table on the official pass: **15.7% < 30% → stop MTP**, keep the experiment branch. KV changed 87/159 drafts versus the 1×1 control, so the cache is not a no-op; it just does not create a usable top-1 rate. Later steps did not improve as the cache grew.

`hidden` after final RMSNorm is the correct MTP input, not a residual risk. Missing 27-token prompt MTP prefill is an incomplete reproduction, but later-window cache already exceeded 100 tokens and last-32 top-1 was about 9%, so it does not overturn the stop line.

Do not write “Qwen3.8 native MTP accept rate is 15.7%.” Write the adapter-bound Gate 5 conclusion in `docs/MTP_GATE5_FINAL.md`.

8K was restored (`:1111=200`, chat `OK`). 400K stayed `:1112=200`. No fused verify, no K=2, no Codex. Smaller Qwen3.8 instances are also abandoned.

## Answers

1. **K=1 是否真实接受 draft？** 否。服务路径只比较 draft top-1 与下一 greedy token，没有 commit/rollback。
2. **14.5% 是什么？** 9/62 的 draft top-1 命中率，不是 scheduler 接受率。
3. **T=0 是否逐 token 一致？** 同一 prompt、64 completion 的可见文本 SHA 与 MTP=0 **完全一致**。这只证明 hook 没改坏 target。
4. **拒绝后 GDN/KV 是否正确？** Target 从不消费 draft，所以 KV 不会多写错误 token。这不是 fused verify 的无损证明。
5. **是否比基线更快？** **这个 hook 更慢。** 64 token 8.6s → 294s。不能外推 GPU MTP。
6. **提升/开销来自哪？** 无加速。开销是每步 CPU fp32 MTP（含 lm_head）+ profile/hook。
7. **是否值得进 K=2？** **否。** 先做 Gate 5 离线 KV 回放。
8. **8K/400K？** 8K 已恢复 200/`OK`；400K 全程 200/`OK`。不接 MTP。

## Hard rules

- 400K / GPU4–7 / :1112 never touched
- 8K container is stopped, not removed, and restored after
