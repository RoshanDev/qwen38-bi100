# 交互延迟 DeepResearch：本机 Qwen3.8 vs ChatGPT / Claude / Grok

日期：2026-08-17  
范围：只做方案，不改线上 8K/400K 服务、不换引擎、不升级宿主驱动。

## 结论

1. 现在慢是真的，而且是两段都慢：Codex 首轮 **TTFT 约 15–60 秒**，出字 **约 7.6 tok/s**。
2. 和云端非推理 API 比，出字大约慢 **8–18 倍**；和同尺寸 Qwen3.6-27B 在 A100/H100 上的公开数字比，大约慢 **8–12 倍**。
3. **现在不要改 SGLang。** 天数 BI-V100 / CoreX 3.2.x 没有可核验的 SGLang 适配；SGLang 的收益依赖 FlashInfer、CUDA Graph、torch.compile，这三条在本机都未打通，其中 CUDA Graph 已有挂死记录。
4. 可感知加速应按风险从低到高做：先砍 Codex 首轮提示和开 prefix cache，再评估 decode 图/量化/MTP。这些都要隔离容器，不能动 GPU0–3 的 8K 回滚。

## 指标怎么比

| 指标 | 含义 | 注意 |
|---|---|---|
| TTFT | 发出请求到第一个可见输出 token | 推理模型会把思考时间算进去，不能和本机 `enable_thinking=false` 直接比 |
| Decode tok/s | 第一个 token 之后的生成速度 | 单并发、固定输出长度才有可比性 |
| Prefill tok/s | 吃输入的速度 | Codex 首轮 9k–28k tokens，TTFT 几乎等于 prefill |
| E2E | 整段墙钟 | 含网络、bridge、tokenize |

云厂商数字来自公开聚合站，不是本机复测；不同文章口径不一致，下表只取能对上定义的值。

## 本机实测（已有，不是估算）

来源：阶段 16 A/B、400K 服务日志、2026-08-17 手动 Codex。

| 场景 | 数字 | 说明 |
|---|---:|---|
| 短 decode，256 tokens | **7.3–7.6 tok/s** | factor2/CHUNK，思考关闭 |
| 8K 服务 512-token e2e | **7.985 tok/s** | 含预热 |
| 16K 冷 prefill | **26.2 s** / 15,971 tokens | ≈ 610 prompt tok/s |
| 64K 冷 prefill | **135 s** / 63,953 tokens | CHUNK 比关 fastpath 快 8.12% |
| Codex 短答 | **13.2 s** / 9,277 input | 2026-08-17 exec |
| Codex 真实交互一轮 | **35–62 s** / 16k–28k input | 本机 bridge `request_ok` |
| 服务端 decode 日志 | **7.5–7.7 tok/s** | `Running: 1 reqs` |
| 服务端 prefill 日志 | **400–1500 prompt tok/s** | 有请求时 GPU4–7 利用率 85–97% |

社区同栈 Qwen3.6-27B / 4×BI-V100 约 **8.9 tok/s**。本机和社区基线同量级，不是这次配错。

Codex 体感拆开看：

- **首 token 慢**：系统提示 + skills 先 prefill 1–3 万 tokens。按 700 tok/s 算，1.6 万 tokens 就要约 23 秒才开始出字。
- **出字慢**：开始生成后固定约 7.6 tok/s。打 80 个汉字大约 10 秒。
- **工具轮次更慢**：每次工具结果再 prefill 一次，28k 输入一轮超过 1 分钟。

## 云端对照（2026-08 公开数据）

### 非推理 / 低思考：和本机关思考更接近

| 模型 | TTFT | Decode | 来源 |
|---|---:|---:|---|
| GPT-4o | 0.45–0.83 s | 100–141 tok/s | EdenAI / BenchLM |
| GPT-5.2 | 0.85 s | 62 tok/s | crazyrouter 2026 汇总 |
| Claude Sonnet 4.5 | 0.58 s | 98 tok/s | 同上 |
| Claude Haiku 4.5 | 0.18–0.60 s | 180 tok/s | crazyrouter / Kunal |
| Grok 4.1 Fast | 0.45–0.54 s | 95–138 tok/s | crazyrouter / BenchLM |
| GPT-5.6 Luna (low) | 1.44 s | 167 tok/s | Artificial Analysis |

### 高思考：TTFT 会被思考吃掉，只比出字

| 模型 | TTFT（含思考） | Decode | 来源 |
|---|---:|---:|---|
| GPT-5.6 Sol (high) | 10.7 s | 70 tok/s | Artificial Analysis |
| GPT-5.6 Sol (max) | 136 s | 74 tok/s | 同上 |
| Claude Opus 4.8 | — | 60 tok/s | AA vs Sol |
| Claude Opus 5 (max) | — | 57 tok/s | AA |
| Grok 4.5 (high) | 8.7–17 s | 56–91 tok/s | AA / CodingFleet |
| Grok 4.6 (high) | 32–36 s | 58–86 tok/s | AA |

### 同尺寸开源，NVIDIA 上

| 设置 | Decode | 来源 |
|---|---:|---|
| Qwen3.6-27B INT8 / A100 | 65–80 tok/s | Reddit 用户报告 |
| Qwen3.6-27B FP8 / A100，关 FP8 KV | 75–100 tok/s | 同上 |
| Qwen3.6 FP8 / RTX Pro 6000 | 80–208 tok/s | 视 MTP 而定 |
| 本机 4×BI-V100 BF16 eager TP4 | **7.6 tok/s** | 本仓库实测 |

和云端非推理 API 比，出字大约 **8–18×**；和 A100 上同代 27B 比，大约 **8–12×**。差距主要是硬件带宽 + 无 CUDA Graph + 无 MTP/量化，不是 Codex 配错窗口。

## 为什么本机快不起来

按对体感的贡献排序：

1. **Decode 受内存带宽和 kernel 启动限制。** 27B 稠密模型每个 token 都要过全部权重。TP=4 每步都要通信。`--enforce-eager` 关掉 CUDA Graph 后，小 kernel 启动开销打在逐步 decode 上。社区已排除 custom all-reduce（单并发无收益），CUDA Graph 在 CoreX+xFormers+TP4 上有卡死记录。
2. **Codex 首轮输入太大。** 9k–28k tokens 的系统提示/工具 schema/skills，TTFT 被 prefill 主导。云端 Codex 通常有前缀缓存，第二次几乎不重算这段。
3. **没有 MTP / speculative decode。** 官方 Qwen3.8 有 MTP，社区文本适配明确关掉。这是把 8 tok/s 拉到 20–30 的正路，但不是改配置。
4. **没有可用的 FP8/INT8 解码核。** NVIDIA 上同尺寸模型靠量化吃带宽。CoreX 3.2.3 这条 Qwen3.5 路径没有验证过对应核。
5. **PREFIX_FLASH / 标准 prefix cache 没开。** 前者有挂死历史，默认镜像未带；启动脚本也没加 `--enable-prefix-caching`。

已经试过、证明没用或很弱的：

| 手段 | 结果 |
|---|---|
| LOOP1 native GDN | 16K 无收益 |
| native RMSNorm | decode 噪声级 |
| `max_num_batched_tokens=8192` | 比 4096 更慢，KV 更少 |
| custom all-reduce | 单并发更慢 |
| CUDA Graph | 社区记录卡死 |
| 只加大 context / 显存利用率 | 解决容量，不加速 decode |

## 能不能改 SGLang

**现阶段不建议。**

| 问题 | 事实 |
|---|---|
| 硬件支持 | 公开资料里天数大模型走自家 vLLM 分支、ixRT、IGIE。MinerU 的 Iluvatar 说明也是 vLLM，而且是更新的 BI-V150 / 驱动 4.4。没有 BI-V100 + CoreX 3.2.3 的 SGLang 配方。 |
| 模型支持 | 官方 SGLang 要较新的 Transformers / FlashInfer。Qwen3.8 混合 GDN + 线性注意力是社区补丁进 vLLM 0.6.3 的，不是 SGLang 现成模型。 |
| 收益前提 | SGLang 相对 vLLM 的常见 10–30% 来自 RadixAttention、CUDA Graph、FlashInfer。本机 Graph 已知风险，FlashInfer 是 NVIDIA 路径。 |
| 工作量 | 等于再做一遍引擎适配：算子、paged attn、TP、GDN、YaRN、工具解析。风险高于继续在已跑通的 vLLM 上挖。 |
| 和 ixkb 的关系 | 用户之前的 MR100/BI150 + CoreX 4.4 教程不能当 BI-V100 方案。 |

只有出现下面任一条件，才值得重开 SGLang 评估：天数提供 CoreX 3.2/3.2.3 或 4.x 的 SGLang 镜像且含 Qwen3.5/3.8；或社区给出可审计的 BI-V100 SGLang + GDN 提交。

## 优化方案（按推荐顺序，全部隔离验收）

边界不变：GPU0–3 的 8K 服务不动；实验只用 GPU4–7 的独立容器；一次只改一个变量；正确性和 tok/s 一起报。

### P0：不换引擎，先降 Codex 体感

1. **Prefix cache**  
   在独立 400K 容器加 vLLM 0.6.3 `--enable-prefix-caching`。验收：同一 Codex 系统提示连续两轮，第二轮 TTFT 是否明显下降。风险：旧 vLLM + GDN 状态缓存可能不正确，必须用双口令回归。  
   预期：第二轮及以后 TTFT 从 20–40s 降到数秒级。对**第一轮冷启动**和 **decode tok/s** 几乎没帮助。

2. **缩短首轮提示**  
   隔离 `CODEX_HOME` 再关一批 skills/插件，或临时打开已有的 `QWEN_COMPACT_CODEX_INSTRUCTIONS`。验收：`estimated_input_tokens` 从 16k–28k 降到多少，TTFT 是否同比下降。  
   预期：线性减少 prefill。不提升出字速度。

3. **交互走短窗口，长活再切 400K**  
   8K 服务已在 GPU0–3。短问答 TTFT 会好一截，但 Codex 完整系统提示本身就接近/超过 8K，所以这条只适合确认“短请求是不是也只有 8 tok/s”。  
   预期：证明 decode 瓶颈与 400K/YaRN 无关。

### P1：还在现有 vLLM 里挖 decode

4. **只对 decode 试 CUDA Graph**  
   社区是整路径卡死。若要试，必须独立容器、可秒级杀掉、先短输出。失败就停。  
   预期：若能稳住，单并发 decode 有机会到 12–20 tok/s。失败模式是挂死。

5. **PREFIX_FLASH / GDN prefix state**  
   社区有提交，本仓库因挂死风险未默认打开。只对 Codex 重复前缀做隔离 A/B。  
   预期：改善 prefill，不解决 7.6 tok/s。

6. **量化**  
   先静态查 CoreX 3.2.3 镜像有没有 FP8/INT8 GEMM 和对应 vLLM 权重量化。没有核就不要转权重量。  
   预期：若核可用，decode 可能接近 A100 公开量级的一半到同级；这是猜测，必须测。

### P2：架构级，不承诺工期

7. **MTP / speculative decode**  
   官方权重有 MTP，当前 loader 是 CausalLM、无视觉/无 MTP。要移植模型类和 draft 路径。  
   预期：这是文献上把逐步 decode 拉到 20–30+ tok/s 的正路。工作量最大。

8. **SGLang**  
   仅在天数或社区给出 BI-V100 可运行配方后立项。现在做是重写运行时。

9. **换 NVIDIA 机器跑官方 vLLM/SGLang + FP8**  
   若目标是“接近 ChatGPT 体感”，这是唯一能在数天内对齐 60–100 tok/s 的办法。不在当前 BI-V100 节点的风险边界内。

## 建议的验收口径

以后任何加速声明必须带这四行，缺一不可：

```text
prompt_tokens=
completion_tokens=
ttft_ms=
decode_tok_s=   # completion / (e2e - ttft)
correctness=    # 短答标记或双口令
```

对照集固定为：

- 短：32 输入 / 256 输出，关思考
- 中：Codex 真实系统提示（约 9k–16k）/ 64 输出
- 前缀：同一系统提示连打两轮（测 cache）
- 正确：16K 双口令

不要用 399K 冷检索冒充交互速度。

## 明确不做

- 不把线上 `qwen38-bi100-400k` 直接改成 SGLang
- 不在共享节点升级驱动、重启 Docker
- 不把“能启动 1M”说成交互变快
- 不拿推理模型 30 秒 TTFT 给本机 7.6 tok/s 开脱——出字慢是另一回事

## 下一步（等你点头再动手）

只建议先做 **P0-1 prefix cache** 和 **P0-2 缩提示** 两件隔离实验。预计 1–2 个工作单元能给出“第二轮 TTFT 能不能从 30s 掉到 5s 以内”的答案。decode 从 7.6 拉到 20+ 没有低风险开关。
