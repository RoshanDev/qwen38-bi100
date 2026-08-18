# MTP Gate 5 最终结论

分支：`exp/mtp-k1-vllm`（不合并运行时代码到 `main`）  
原始日志：`/data/qwen38/logs/mtp-gate5-kv/`  
仓库只保留哈希、统计和复现命令。清单见 [../artifacts/mtp-gate5-manifest.json](../artifacts/mtp-gate5-manifest.json)。

## 不要这样写

> Qwen3.8 自带 MTP 的真实接受率只有 15.7%。

那会把当前适配器上的离线回放，误写成模型固有能力。

## 应该这样写

在当前 CoreX 3.2.3 / vLLM 0.6.3 适配器上，使用真实 target positions、正确 post-final-norm hidden、Gemma RMSNorm `(1+w)` 和持续维护的一层 MTP KV 进行离线回放，159 步 top-1 命中率为 15.7%。虽然未对 27-token prompt 建立 MTP prefill KV，但后半段缓存超过 100 tokens 后命中率仍仅约 9%，不足以支持继续实现 GPU MTP 和 fused target verification。因此按预设 `<30%` 止损线停止该路线。

## 生产决策

| 项目 | 定论 |
| --- | --- |
| 8K 回滚服务 | 保留 `text-0e899`，不替换 |
| 400K 生产服务 | 保持 `dense-native-v1`，不接 MTP |
| dense-native | TTFT 有收益，decode 无收益 |
| MTP 权重与结构 | 已确认可加载，Gemma RMSNorm 已校正 |
| CPU observer | 仅用于正确性观察，不代表性能 |
| Gate 5 KV 回放 | 真实 position、有 MTP KV，top-1 15.7% |
| fused verify | 不实现 |
| K=2 | 不进行 |
| Codex 接入 MTP | 不进行 |
| 更小的 Qwen3.8 实例 | 放弃。效果不如当前 BF16 TP4 27B |
| 实验分支 | 保留，作为证据和后续参考 |

## 为什么 14.5% / 294s 不能当最终 MTP 结论

Observer hook 每次只构造当前 token 的 Q/K/V，一层 `full_attention` 退化成 1×1 softmax；position 用进程全局计数器，不是 vLLM `positions`；也没有 commit/rollback。它只证明「这个 CPU 无状态观察器不加速」，不能证明「Qwen3.8 自带 MTP 在 BI-V100 上没有加速价值」。

64-token 文本 SHA 与 MTP=0 一致，只说明 hook 没改坏 target 输出。`8.6s → 294s` 是 rank0 CPU FP32 Linear + 248K `lm_head` 的观察器成本。

## Gate 5 做了什么

1. 从一次确定性 target 生成记录 160 个 greedy token，以及每步 last-layer hidden、vLLM 真实 absolute position、下一 greedy id。
2. 离线顺序回放 MTP：`sampled token embedding + preceding target hidden`，position 只读记录值。
3. 正式配置维护一层 MTP Full Attention KV：每步追加 K/V，Query 对全部历史做 causal attention。
4. 同时输出无 KV 对照，以及 position `±1/±2` 诊断（非正式配置）。

## 数字

Prompt：`用两段中文解释矩阵乘法为什么是 Transformer 的主要计算，不要提纲。`  
T=0，关思考，`finish_reason=length`。tokenizer 长度 28。记录 position 27…185，每步严格 +1。

| 配置 | top-1 | top-5 | mean rank | median rank |
| --- | ---: | ---: | ---: | ---: |
| 正式：MTP KV + 记录 position | **15.7%**（25/159） | 44.7% | 87.0 | 9 |
| 对照：无 KV，1×1 | 16.4%（26/159） | 40.3% | 106.3 | 9 |
| 诊断 −2 / −1 / +1 / +2（前 32 步，开 KV） | 12.5% / 15.6% / 12.5% / 12.5% | ~41% | 更差 | — |

补充：

- `lm_head(hidden) == sampled`：**159 / 159**
- KV 改写了 87/159 个 draft，不是空操作
- 前 / 中 / 后 32 步 top-1：12.5% / 15.6% / 9.4%
- 后半段缓存已超过 100 tokens

止损表：`<30%` 停止 MTP；`30%～60%` 先对齐；`60%～75%` 才考虑 GPU K=1；`>75%` 才做 GPU MTP + 一次 target forward 验两个 token。正式结果落在 `<30%`。

## 两项曾经写成「残留风险」的点

### 1. hidden 取 final norm 之后：这是正确的，不是风险

上游 Qwen3.5 / Qwen3.8 主模型在返回用于 `lm_head` 的 hidden 之前会做最终 RMSNorm。官方 MTP predictor 接收的就是这份 target hidden，然后再执行自己的 `pre_fc_norm_hidden`，与 sampled-token embedding 拼接后进入 `mtp.fc`。

`lm_head(hidden) == sampled` 159/159 也说明 dump 到的是正确的、`lm_head` 前的最终 hidden，不是错误的中间层输出。

### 2. 没有对 prompt 前 27 个位置做 MTP prefill：未完全复现，但不足以推翻止损

官方 MTP decoder 是一层 `full_attention`，严格复现应给它建立 prompt 阶段的 MTP KV。

但数据已经不利于「补完 prompt KV 后会突然变好」：

- prompt 只有 27 tokens
- 后半段 MTP KV 已超过 100 tokens
- 后半段 top-1 仍约 9%
- 加 KV 后 top-1 为 15.7%，无 KV 反而是 16.4%
- KV 改写了 87/159 个 draft，却没有提高总体命中
- position `±1/±2` 都没有改善

不能从数学上证明「补 prompt KV 一定没提升」，但没有工程证据支持它能把 15.7% 拉到值得实现 fused verify 的 60%～80%。为这一个可能性继续停机、补 prompt hidden dump 和重放，投入产出不成立。

## 复现

```bash
# GPU0-3 临时停 8K（保留容器），dump 后自动恢复，再 CPU 回放
# 绝不碰 400K / GPU4-7
bash /data/qwen38/scripts/run_mtp_gate5_kv.sh

# 已有 dump 时，只做 CPU 回放
python3 scripts/mtp_gate5_replay.py \
  --model /data/qwen38/models/Qwen3.8-27B \
  --dump /data/qwen38/logs/mtp-gate5-kv/mtp-gate5-dump.pt \
  --out /data/qwen38/logs/mtp-gate5-kv/mtp-gate5-replay.json
```

本地单测（不需要模型权重）：

```bash
python3 -m unittest tests.test_mtp_gate5_stats -v
```

## 更大范围的闭环

窗口、dense-native decode、FP8、MTP 四条优化方向都已经有证据。更小的 Qwen3.8 实例效果不如当前 BF16 TP4 27B，也放弃。继续追求 70～80 tok/s，应转向真正的 BI 低比特内核或其他硬件，不再继续挤当前这套 27B 服务。
