# Qwen3.8-27B on Iluvatar BI-V100

公开仓库：<https://github.com/RoshanDev/qwen38-bi100>

## 已验证状态

2026-08-16 已在一台 8×BI-V100 32GB 服务器上实测部署成功。8K 回滚服务与 400K 长上下文服务当前同时运行。

| 项目 | 8K 回滚服务 | 400K 主服务 |
|---|---|---|
| API | `http://QWEN_HOST:1111/v1` | `http://QWEN_HOST:1112/v1` |
| 容器 | `qwen38-bi100-server` | `qwen38-bi100-400k` |
| 镜像 | `qwen38-bi100:corex3.2.3-text-0e899` | `qwen38-bi100:corex3.2.3-dense-native-v1` |
| GPU | 0–3，TP=4 | 4–7，TP=4 |
| 上下文 | 8192 | 400000，YaRN factor=2 |
| 关键参数 | eager、显存利用率 0.80 | eager、chunked prefill=4096、CHUNK fastpath、显存利用率 0.95 |

模型固定在官方 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。18 个权重分片、1199 个索引键和总字节均已离线校验。

长上下文不是只验证了启动参数。双口令分别埋在提示词约 10% 和 90% 位置，实际结果如下：

| 实际 prompt tokens | 端到端耗时 | 检索结果 |
|---:|---:|---|
| 15,971 | 26.684s | 通过 |
| 31,965 | 57.354s | 通过 |
| 63,953 | 142.991s | 通过 |
| 95,963 | 255.270s | 通过 |
| 271,965 | 1,350.743s | 通过 |
| 398,971 | 2,675.417s | 通过 |

因此 Codex 使用 400K context、390K bridge 输入安全线和 360K 自动压缩阈值，最多为输出保留 8192 tokens。398,971-token 冷请求需约 44 分 35 秒，400K 表示容量与正确性，不表示可交互的冷启动延迟。

1M/factor4 服务也已实机启动：65,485 GPU blocks、1,047,760-token KV 容量、1M 请求并发 1.05x，并通过 16K 正确性回归。但未执行接近 1M 的完整提示，因此不把它写成“1M 全长已验证”。

## 调用示例

如果客户端设置了全局代理，访问内网地址时需要绕过代理：

```bash
curl --noproxy '*' -sS \
  http://QWEN_HOST:1112/v1/chat/completions \
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

当前旧版 vLLM 不会把默认 `<think>` 内容拆成独立的 reasoning 字段。短回答建议传入 `chat_template_kwargs.enable_thinking=false`；否则较小的 `max_tokens` 可能只容纳思考过程。

## 日常运维

查看状态和日志：

```bash
ssh root@QWEN_HOST
docker ps --filter name=qwen38-bi100-server
docker logs -f --tail=200 qwen38-bi100-server
docker ps --filter name=qwen38-bi100-400k
docker logs -f --tail=200 qwen38-bi100-400k
```

停止和重新启动现有容器：

```bash
docker stop qwen38-bi100-server
docker start qwen38-bi100-server
```

如果容器已被删除，可用已固化脚本重新创建：

```bash
bash /data/qwen38/scripts/start_server.sh
```

重建 400K 服务：

```bash
bash /data/qwen38/deployment/qwen38-bi100/scripts/start_long_context_server.sh
```

重新执行正确性测试和基准：

```bash
python3 /data/qwen38/scripts/smoke_api.py --wait-seconds 60
python3 /data/qwen38/scripts/benchmark_api.py --max-tokens 512
python3 /data/qwen38/deployment/qwen38-bi100/scripts/test_long_context.py \
  --base-url http://127.0.0.1:1112 \
  --target-input-tokens 272000
```

## 校验与复现

模型的官方配置备份位于：

```text
/data/qwen38/models/Qwen3.8-27B/config.json.qwen38-official
```

当前 `config.json` 只将入口从 `Qwen3_5ForConditionalGeneration` 改成文本适配器 `Qwen3_5ForCausalLM`。重新校验：

```bash
docker run --rm \
  -v /data/qwen38/models/Qwen3.8-27B:/model:ro \
  -v /data/qwen38/scripts:/scripts:ro \
  --entrypoint python3 \
  qwen38-bi100:corex3.2.3-text-0e899 \
  /scripts/validate_model.py /model
```

基础镜像归档保留在：

```text
/data/qwen38/images/bi100-3.2.3.zte-x86-ubuntu20.04-py3.10-poc-llm-infer-v1.2.3.tar.gz
```

其 SHA-256 为：

```text
c32d96d96f7c6e81e45d7fa530e00ab86450deb5cc3ca00fe4adb9472e73d504
```

重新构建派生镜像：

```bash
BASE_IMAGE='YOUR_AUTHORIZED_COREX_3_2_3_LLM_IMAGE:v1.2.3' \
BUILD_ROOT=/data/qwen38/build \
bash scripts/build_image.sh
```

构建脚本会临时获取并严格校验社区适配提交 `0e899064...`，不会把第三方补丁源码复制进本仓库。CoreX 基础镜像需通过你有权使用的天数渠道自行取得并预先导入本机。

长上下文镜像使用最后一个 dense 适配提交 `d972854...` 做增量构建，不需要重新联网安装依赖：

```bash
COREX_BASE_IMAGE='YOUR_AUTHORIZED_COREX_3_2_3_LLM_IMAGE:v1.2.3' \
QWEN38_BASE_IMAGE='qwen38-bi100:corex3.2.3-text-0e899' \
bash scripts/build_long_context_image.sh
```

再构建默认关闭、可独立 A/B 的 dense native fastpath 镜像：

```bash
BASE_IMAGE='qwen38-bi100:corex3.2.3-longctx-d972' \
bash scripts/build_dense_native_image.sh
```

400K overlay 使用硬链接复用官方权重，不重复占用约56GB：

```bash
python3 scripts/prepare_yarn_model.py \
  /data/qwen38/models/Qwen3.8-27B \
  /data/qwen38/models/Qwen3.8-27B-YaRN-400K-kvfix \
  --factor 2.0 \
  --target-context 400000
```

详细原理、1M 探针与验收步骤见 [长上下文适配与实测](docs/LONG_CONTEXT.md)。

## 已知边界

- 当前适配仅验证文本推理，不支持图片输入。
- 未启用 MTP 推测解码、FP8 或 CUDA graph。
- 400K 服务启用了社区 dense chunked-prefill/paged-attention 路径和本项目 CHUNK recurrence fastpath；当前最高真实检索为 398,971 prompt tokens。
- 旧 vLLM 将 64 层混合模型误算成 64 个 KV 层，导致旧探针只有 261,936-token 容量。本项目给 overlay 写入顶层 `layers_block_type` 后，vLLM 正确识别 16 个 full-attention 层，1M/factor4 服务容量达到 1,047,760 tokens。
- 1M 只完成容量启动和 16K 回归；接近 1M 的冷 prefill 预计需要数小时，未做全长正确性声明。
- 生成速度仍约 7.3–7.6 tok/s。CHUNK 对同轮 64K 冷 prefill 提升 8.12%；LOOP1、RMSNorm、8192 chunk 均未显示足够稳定的收益，CUDA Graph 在该社区路径有 OOM/挂死记录，默认不启用。
- 经过接近 7K 输入的 Codex 长请求后，CoreX 缓存的工作区会使 GPU0–3 显存升至约 30GB；0.80 的启动参数已为 eager SDPA 留出临时空间。
- CoreX 基础镜像未随仓库分发，也没有发布到 Docker Hub；当前没有找到足以证明该私有运行时可公开再分发的授权文本。

## Codex CLI

仓库提供 Responses → Chat Completions 本地 bridge 和完全隔离的 Codex 配置。它不会覆盖 `~/.codex/config.toml`；OpenAI 官方 Codex 继续使用 `codex`，Qwen 入口使用 `codex-qwen38`。

```bash
bash scripts/install_codex_integration.sh http://QWEN_HOST:1112/v1
codex-qwen38 exec --ephemeral --skip-git-repo-check '只回答 CODEX_QWEN_OK'
```

单行执行时不要附加反斜杠；多行命令只能在每个待续行末尾使用一个 `\`。bridge 会用上游 tokenizer 精确计算 token 预算，并在 390K 输入安全线内保留最多 8192 输出 tokens。

安装、普通回复测试、真实工具调用测试及卸载方法见 [Codex CLI 接入与手动验收](docs/CODEX.md)。

## 来源

- [Qwen3.8-27B 官方模型](https://huggingface.co/Qwen/Qwen3.8-27B)
- [BI-V100 社区适配提交](https://dev.modelhub.org.cn/icer/qwen36_01/commit/0e89906481e9a6cb2475925adb41517676e12903)
- [BI-V100 dense 100K 提交](https://dev.modelhub.org.cn/icer/qwen36_01/commit/d972854fb79f47aa6ab1b9a5a45d27ca99c8c5ab)
- [社区四卡运行记录](https://dev.modelhub.org.cn/icer/qwen36_01/src/branch/main/worklogs/2026-07-13-initial-run.md)
- [用户提供的天数知识库参考页](https://ixkb.iluvatar.com.cn:9443/webdoc/view/Pub8a16948a9a4cb023019cbca37f861590.html)
