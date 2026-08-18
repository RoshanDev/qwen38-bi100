# Qwen3.8-27B BI-V100 安装部署、400K 上下文与 9.11 灾备

适用环境：推理机 `root@192.168.9.20`（8 x Iluvatar BI-V100 32GB，宿主 CoreX 3.2.0），备份机 `fanghui@192.168.9.11`，本机 WSL 通过隔离命令 `codex-qwen38` 接入。公开源码：https://github.com/RoshanDev/qwen38-bi100 （`main` 含 400K KV 修复与 Codex catalog；MTP 证据在实验分支 `exp/mtp-k1-vllm`，不合并运行时代码）。

本文只写已经跑通的路径。不要升级 9.20 宿主驱动，不要重启 Docker，不要把 56GB 权重或 CoreX 镜像写进 WSL / Docker Desktop 磁盘。

## 最终定论（2026-08-18）

生产保持 8K 回滚 + 400K 主服务。窗口、dense-native decode、FP8、MTP 四条优化方向都已经有证据闭环。更小的 Qwen3.8 实例效果不如当前 BF16 TP4 27B，也放弃。继续追求 70～80 tok/s，应转向真正的 BI 低比特内核或其他硬件，不再继续挤当前这套 27B 服务。

不要写：「Qwen3.8 自带 MTP 的真实接受率只有 15.7%。」

更准确的是：

> 在当前 CoreX 3.2.3 / vLLM 0.6.3 适配器上，使用真实 target positions、正确 post-final-norm hidden、Gemma RMSNorm `(1+w)` 和持续维护的一层 MTP KV 进行离线回放，159 步 top-1 命中率为 15.7%。虽然未对 27-token prompt 建立 MTP prefill KV，但后半段缓存超过 100 tokens 后命中率仍仅约 9%，不足以支持继续实现 GPU MTP 和 fused target verification。因此按预设 `<30%` 止损线停止该路线。

| 项目 | 定论 |
| --- | --- |
| 8K 回滚服务 | 保留旧镜像 `text-0e899`，不替换 |
| 400K 生产服务 | 保持 `dense-native-v1`，不接 MTP |
| dense-native | TTFT 有收益（约 0.45s → 0.25s），decode 无收益 |
| FP8 | 官方 128×128 block-FP8，当前 CoreX vLLM 无对应融合核；不下 30.9GB 权重 |
| MTP 权重与结构 | 已确认可加载，Gemma RMSNorm `(1+w)` 已校正 |
| CPU observer | 仅用于正确性观察。14.5% 不是接受率，294s 不是 MTP 性能 |
| Gate 5 KV 回放 | 真实 position、有 MTP KV，top-1 15.7% |
| fused verify | 不实现 |
| K=2 | 不进行 |
| Codex 接入 MTP | 不进行 |
| 更小的 Qwen3.8 实例 | 放弃 |
| 实验分支 | 保留 `exp/mtp-k1-vllm` |

## 当前线上状态

两套服务同时在线，模型名都是 `Qwen3.8-27B`，权重 revision 固定为 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。

| 用途 | 容器 | 端口 | GPU | 镜像 | 上下文 |
| --- | --- | ---: | --- | --- | --- |
| 8K 回滚 | `qwen38-bi100-server` | 1111 | 0-3 | `qwen38-bi100:corex3.2.3-text-0e899` | 8192 |
| 400K 主服务 | `qwen38-bi100-400k` | 1112 | 4-7 | `qwen38-bi100:corex3.2.3-dense-native-v1` | 400000，YaRN factor=2 |

日常交互和 Codex 走 **1112 / 400K**。1111 只作回滚。访问内网必须加 `--noproxy '*'`，否则会被本机 HTTP 代理误送。

```
curl --noproxy '*' -fsS http://192.168.9.20:1112/health
curl --noproxy '*' -fsS http://192.168.9.20:1112/v1/models
```

`/v1/models` 里应看到 `id=Qwen3.8-27B` 且 `max_model_len=400000`。

## 优化路线证据

### 1. 窗口：100K / 200K / 400K 都不改 decode 瓶颈

同镜像、同 CHUNK、只改 `max-model-len`。短生成 decode：8K 8.15、400K 7.52、200K 7.48、100K 7.36 tok/s。热启动 TTFT 约 0.45–0.57s。压窗口不能到 70–80 tok/s。

400K 表示容量和正确性。双口令分别埋在约 10% / 90%：

| 实际 prompt tokens | 端到端耗时 | 检索结果 |
| ---: | ---: | --- |
| 15,971 | 26.684s | 通过 |
| 31,965 | 57.354s | 通过 |
| 63,953 | 142.991s | 通过 |
| 95,963 | 255.270s | 通过 |
| 271,965 | 1,350.743s | 通过 |
| 398,971 | 2,675.417s | 通过 |

1M/factor4 能启动（65,485 blocks / 1,047,760-token KV），并通过 16K 回归。没有做接近 1M 的全长正确性，不要当日常入口。

根因：vLLM 0.6.3 会把 64 层混合模型误算成 64 个 KV 层。400K overlay 已写入顶层 `layers_block_type`，按 16 个 full-attention 层分配。这解决容量，不解决出字速度。

### 2. dense-native：TTFT 有收益，decode 无收益

同一 BF16、GPU0–3、TP=4、T=0、关思考。8K 回滚镜像是 `text-0e899`；A–D 是 `dense-native-v1`。400K 全程未动。结束后 8K 已恢复。

| 变体 | 256 TTFT / decode | 1024 TTFT / decode |
| --- | --- | --- |
| 8K prod eager | 0.454s / **8.174** | 0.462s / **8.001** |
| A native flags 全关 + eager | 0.481s / 7.635 | 0.470s / 7.653 |
| B LOOP1+CHUNK+RMS + eager | **0.254s** / 7.844 | 0.253s / 7.870 |
| C LOOP1+CHUNK+RMS 无 eager | **0.251s** / 7.905 | **0.251s** / 7.889 |
| D 同 C + custom all-reduce | 0.255s / 7.669 | 0.251s / 7.940 |

没有一个变体 decode 超过 8K 基线。C/D 去掉 `--enforce-eager` 能启动，没有社区记录的 Graph 挂死。8K 保持 `text-0e899`。400K 可继续用已在线的 CHUNK-only native。

### 3. FP8：当前栈跑不了官方 block-FP8

官方 Hugging Face 与 ModelScope 的 `Qwen/Qwen3.8-27B-FP8` 是同一 checkpoint：`config.json` / `index.json` SHA-256 一致。量化是 `fp8` + `e4m3` + dynamic + `weight_block_size=[128,128]`。当前 CoreX vLLM 0.6.3 只有 per-tensor FP8 / NVIDIA cutlass-marlin，没有 BI-V100 的 128×128 融合核。不下 30.9GB 权重，也不走社区 Q8 GGUF。

### 4. MTP：Gate 5 后正式止损

Observer hook 的 14.5% 只是 draft top-1 命中率，294s 是 CPU 无状态观察器成本。它没有 MTP KV、不用真实 position、不 commit/rollback。

Gate 5 离线回放补上了真实 vLLM positions 和一层 MTP Full Attention KV：

| 配置 | top-1 | top-5 | median rank |
| --- | ---: | ---: | ---: |
| 正式：KV + 记录 position | **15.7%**（25/159） | 44.7% | 9 |
| 对照：无 KV | 16.4% | 40.3% | 9 |
| position ±1 / ±2（前 32 步） | 12.5%–15.6% | ~41% | 更差 |

`lm_head(hidden)==sampled` 为 159/159。hidden 取 final norm 之后是正确的：官方 MTP 接收的就是这份 target hidden，再做自己的 `pre_fc_norm_hidden`。没有对 27-token prompt 建 MTP prefill KV 是未完全复现，但后半段缓存已超过 100、top-1 仍约 9%，加 KV 后 top-1 还略低于无 KV。没有工程证据支持把它拉到 60%–80%。按 `<30%` 止损。

原始大日志留在 `/data/qwen38/logs/mtp-gate5-kv/`。仓库只保存哈希和统计：`docs/MTP_GATE5_FINAL.md`、`artifacts/mtp-gate5-manifest.json`。

### 5. 更小的 Qwen3.8 实例：放弃

质量不如当前 BF16 TP4 27B。不把它当作加速替代。

## 400K 怎么用

### 直接调 API

关闭思考。旧 vLLM 0.6.3 不会把 `<think>` 拆到独立字段，短 `max_tokens` 可能只吐思考过程。

```
curl --noproxy '*' -sS \
  http://192.168.9.20:1112/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [
      {"role": "user", "content": "用三句话解释 Go 的并发模型。"}
    ],
    "temperature": 0.2,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### Codex 隔离接入（推荐）

不会改 `~/.codex/config.toml`，也不会动官方 `codex` 登录。安装：

```
cd ~/Developer/Qwen3.8-27B
bash scripts/install_codex_integration.sh http://192.168.9.20:1112/v1
```

只写入：

```
~/.local/share/codex-qwen38/   # config.toml, bridge.env, model-catalog.json
~/.config/systemd/user/qwen38-codex-bridge.service
~/.local/bin/codex-qwen38
```

确认 bridge：

```
systemctl --user status qwen38-codex-bridge.service --no-pager
curl --noproxy '*' -fsS http://127.0.0.1:8348/healthz
```

健康检查应返回 `upstream=http://192.168.9.20:1112/v1`。

短答验收：

```
mkdir -p /tmp/qwen38-codex-test
codex-qwen38 exec --strict-config --ephemeral --skip-git-repo-check --ignore-rules \
  -C /tmp/qwen38-codex-test \
  '只回答 CODEX_QWEN_OK，不要调用工具。'
```

预期最后一行是 `CODEX_QWEN_OK`。交互模式直接跑 `codex-qwen38`。官方 Codex 仍用 `codex`。

隔离配置里的 400K 预算：

- Codex `model_context_window=400000`
- 360000 触发自动压缩
- bridge 输入安全线 390000
- 单次输出最多 8192
- `model_catalog_json` 声明本地模型，避免 `Model metadata for Qwen3.8-27B not found`

截图：当前 400K 服务是 text-only，不会加载官方视觉塔。Codex 粘贴图片后，本机 bridge 用 Tesseract（`chi_sim+eng`）转文字再发给 Qwen。

### 怎么确认打到了 9.20 GPU

一次请求应对上三处：

1. 本机：`journalctl --user -u qwen38-codex-bridge.service -f` 出现 `request_ok` 和 `POST /v1/responses 200`
2. 9.20：`docker logs -f --tail=20 qwen38-bi100-400k` 出现来自 `192.168.8.13` 的 `POST /tokenize` 和 `POST /v1/chat/completions 200`，并有 `Running: 1 reqs`
3. 9.20：`LD_LIBRARY_PATH=/usr/local/corex/lib64 /usr/local/corex/bin/ixsmi` 中 **GPU4-7 利用率升高**。GPU0-3 是 8K 回滚，空闲时利用率可以是 0。显存接近 32GB 只说明模型常驻，不证明这次在算。

空闲时 vLLM 会打 `prompt 0.0 / generation 0.0 / Running: 0 / KV 0.0%`，这是心跳，不是挂了。开始出字后 generation 大约 **7.5-7.8 tok/s**。`Pending: 1` 表示上一轮还没完，下一轮在排队（服务 `max-num-seqs=1`）。

### 速度预期

400K 表示容量和正确性，不表示交互很快。

| 场景 | 实测 |
| --- | --- |
| 出字 | 约 7.3-7.7 tok/s |
| 16K 冷 prefill | 约 26 s |
| Codex 首轮（约 9k-16k 系统提示） | 十几秒到一分钟 |
| 398971 token 双口令 | 约 44 分 35 秒，结果正确 |

不要用接近 400K 的冷检索当日常交互。

## 容器停了怎么拉起来

先看还在不在：

```
ssh root@192.168.9.20
docker ps -a --filter name=qwen38-bi100
```

容器还在、只是停了：

```
docker start qwen38-bi100-server
docker start qwen38-bi100-400k
```

容器被删了，用已固化脚本重建。不要两个容器抢同一组 GPU 或同一个端口。

8K 回滚（GPU0-3 / 1111）：

```
bash /data/qwen38/scripts/start_server.sh
```

400K 主服务（GPU4-7 / 1112）：

```
bash /data/qwen38/deployment/qwen38-bi100/scripts/start_long_context_server.sh
```

脚本默认值：镜像 `qwen38-bi100:corex3.2.3-dense-native-v1`，模型目录 `/data/qwen38/models/Qwen3.8-27B-YaRN-400K-kvfix`，`MAX_MODEL_LEN=400000`，`QWEN38_CHUNK_PARALLEL=1`，`enforce-eager`，`max-num-seqs=1`。

短回归：

```
python3 /data/qwen38/scripts/smoke_api.py --wait-seconds 60
python3 /data/qwen38/deployment/qwen38-bi100/scripts/test_long_context.py \
  --base-url http://127.0.0.1:1112 \
  --target-input-tokens 16000
```

## 从零恢复（9.20 上没有镜像时）

CoreX 基础层不在 GitHub / Docker Hub。内部用备份归档 `docker load`，不要去公共 registry 拉。

1. 在 9.11 校验并传回 9.20，或在 9.20 直接 `docker load` 已有归档。
2. 400K overlay 权重是硬链接，不要再复制一份 56GB。源模型在 `/data/qwen38/models/Qwen3.8-27B`，400K 配置在 `/data/qwen38/models/Qwen3.8-27B-YaRN-400K-kvfix`。若 overlay 丢了、权重还在：

```
python3 scripts/prepare_yarn_model.py \
  /data/qwen38/models/Qwen3.8-27B \
  /data/qwen38/models/Qwen3.8-27B-YaRN-400K-kvfix \
  --factor 2.0 \
  --target-context 400000
```

3. 再跑上一节的 `start_long_context_server.sh`。

## 9.11 备份在哪

备份根目录：

`fanghui@192.168.9.11:/home/fanghui/backup/qwen38-bi100-20260816`

清单文件：`MANIFEST.md`。大约 90GB。不要对 `/home/fanghui/backup` 做全量 `du`。

| 路径 | 内容 |
| --- | --- |
| `host-9.20/qwen38/` | 9.20 上的模型、源码、脚本、基础镜像归档 |
| `images/qwen38-bi100-corex3.2.3-text-0e899.tar.gz` | 8K 派生镜像，10,342,902,541 字节，SHA256 `c1f8828c5f09cc034f00ee37bd8e986144f3b0d3ec2653aa396c47b74ddf6a25` |
| `images/qwen38-bi100-corex3.2.3-longctx-d972.tar.gz` | 长上下文中间镜像，SHA256 `fc60069b905530ad640f9eed22a0394f6d44246fa0591038c4979e10784fbaa4` |
| `incremental-400k/images/qwen38-bi100-corex3.2.3-dense-native-v1.tar.gz` | 400K 当前镜像，10,343,137,293 字节，SHA256 `cb55b3e861e99f16ea121813cb0c2653fc1c23baf50e0e112e1d72fffad6e41a` |
| `incremental-400k/host/` | 400K/1M overlay 配置、日志、部署树 |
| `incremental-400k/wsl/` | WSL 仓库副本、隔离 Codex 配置（含 8 月 17 日 catalog / OCR 增量） |

8 月 17 日增量只刷新了脚本、文档和隔离配置，没有再拷镜像。公开 commit：`0a34c88`，已合入 `main`（merge `85ebf5b`）。MTP 止损证据在 `exp/mtp-k1-vllm`，不进 `main`。

恢复镜像（必须在归档所在目录执行 `sha256sum -c`，相对文件名才对得上）：

```
ssh fanghui@192.168.9.11
cd /home/fanghui/backup/qwen38-bi100-20260816/incremental-400k/images
sha256sum -c qwen38-bi100-corex3.2.3-dense-native-v1.tar.gz.sha256
gzip -t qwen38-bi100-corex3.2.3-dense-native-v1.tar.gz
gzip -dc qwen38-bi100-corex3.2.3-dense-native-v1.tar.gz | ssh root@192.168.9.20 docker load
```

WSL Codex 恢复只拷这些，不要拷 history / SQLite / installation ID / 官方 `~/.codex`：

- `config.toml`
- `bridge.env`
- `model-catalog.json`
- `qwen38-codex-bridge.service`
- 仓库工作树

恢复后：

```
systemctl --user daemon-reload
systemctl --user enable --now qwen38-codex-bridge.service
curl --noproxy '*' -fsS http://127.0.0.1:8348/healthz
```

## 已知边界

- 当前适配是文本推理。Codex 截图靠本机 OCR，不是官方视觉塔。
- 未启用 MTP、FP8、CUDA Graph。Graph 在该 CoreX + xFormers + TP4 路径有挂死记录。MTP 已按 Gate 5 `<30%` 止损。
- 出字约 8 tok/s，和社区 Qwen3.6 / 4 x BI-V100 同量级。不要指望改 SGLang 立刻变快；天数这条线没有可核验的 BI-V100 + CoreX 3.2.3 SGLang 配方。
- 旧 vLLM 会把 64 层混合模型误算成 64 个 KV 层。400K overlay 已写入顶层 `layers_block_type`，实际按 16 个 full-attention 层分配。
- CoreX 基础镜像没有公开再分发授权，不要推 Docker Hub。
- 不要再挤当前 BF16 TP4 27B 去追 70～80 tok/s。

## 常用命令速查

```
# 服务
ssh root@192.168.9.20 'docker ps --filter name=qwen38-bi100'
ssh root@192.168.9.20 'docker logs -f --tail=50 qwen38-bi100-400k'
ssh root@192.168.9.20 'LD_LIBRARY_PATH=/usr/local/corex/lib64 /usr/local/corex/bin/ixsmi'

# Codex
journalctl --user -u qwen38-codex-bridge.service -n 50 --no-pager
systemctl --user restart qwen38-codex-bridge.service

# 备份
ssh fanghui@192.168.9.11 'sed -n "1,80p" /home/fanghui/backup/qwen38-bi100-20260816/MANIFEST.md'
```
