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

K=1 in this container is **propose + verify against the next target greedy token**. Rejected drafts are not written into target KV/GDN. That keeps T=0 identity but adds a CPU MTP forward every decode step, so it is slower than baseline.

## Answers

1. **K=1 是否真实接受 draft？** 是。服务路径上 9 次 accept / 53 次 reject。
2. **接受率？** 9/62 = **14.5%**（`zh_explain` 64 token 这一轮）。
3. **T=0 是否逐 token 一致？** 同一 prompt、64 completion 的可见文本 SHA 与 MTP=0 **完全一致**。未跑完 256/1024 全表。
4. **拒绝后 GDN/KV 是否正确？** Target 从不消费未验证 draft，KV 不会多写错误 token。这是保守路径，不是 fused 2-token verify。
5. **是否比基线更快？** **否。** 64 token 8.6s → 294s。
6. **提升/开销来自哪？** 无加速。开销是每步 CPU fp32 MTP（含 lm_head）+ profile/hook。未做 fused target verify。
7. **是否值得进 K=2？** **否。** 合同：K=1 ≤ 基线则停止扩展。
8. **8K/400K？** 8K 已恢复 200/`OK`；400K 全程 200/`OK`。

## Hard rules

- 400K / GPU4–7 / :1112 never touched
- 8K container is stopped, not removed, and restored after
