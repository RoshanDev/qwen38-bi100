# BI-V100 长上下文适配与实测

## 结论

- Qwen3.8-27B 官方配置的原生上下文是 262,144 tokens；官方 model card 给出了静态 YaRN `factor=4.0` 扩展到 1,000,000 tokens 的方法。
- 本项目在 4×BI-V100 32GB、TP=4、CoreX/vLLM 0.6.3 上实际跑通 100,000 上下文，并完成最高 95,963 prompt tokens 的双位置检索。
- 同一环境的 1M YaRN 启动探针只能分配 261,936 tokens 的 KV cache，因此无法启动 1M 服务。Codex 当前配置为 100K，而不是虚报 1M。

官方模型说明：<https://huggingface.co/Qwen/Qwen3.8-27B>

## 社区提交选择

社区 dense 27B 提交链依次加入 20K、paged attention、50K、chunked prefill、100K 和大输入 token 统计修复。本项目固定使用最后一个 dense 提交：

```text
d972854fb79f47aa6ab1b9a5a45d27ca99c8c5ab
```

其后的 `629f878...` 开始转为 MoE，不属于 dense Qwen3.8-27B 路线。

原提交仍有一次 `/tmp/vllm_decode_debug.log` 写入，而且 Qwen3.5 adapter 读取了 `rope_parameters` 却没有向 vLLM `get_rope()` 传递 YaRN。本仓库的 [补丁](../patches/d972854-long-context.patch)只做两项修正：删除调试写入；仅在 `rope_type=yarn` 时传递 scaling 参数。默认 native 路径保持不变。

社区提交：<https://dev.modelhub.org.cn/icer/qwen36_01/commit/d972854fb79f47aa6ab1b9a5a45d27ca99c8c5ab>

## 构建

构建脚本不复制第三方源码进本仓库。它从本地社区仓库提取固定 commit，并以无模糊匹配的 `git apply --check` 应用本项目补丁：

```bash
cd /data/qwen38/deployment/qwen38-bi100

COREX_BASE_IMAGE='YOUR_AUTHORIZED_COREX_3_2_3_LLM_IMAGE:v1.2.3' \
QWEN38_BASE_IMAGE='qwen38-bi100:corex3.2.3-text-0e899' \
ADAPTER_REPO_DIR='/data/qwen38/src/enginex-vllm-bi100-qwen36' \
BUILD_ROOT='/data/qwen38/build' \
bash scripts/build_long_context_image.sh
```

生成镜像：

```text
qwen38-bi100:corex3.2.3-longctx-d972
```

本次实测 image ID 为 `sha256:dc0a97ab3c6494cdf495cf5f075525cb57e3269fc7c5c8ff58c53ad41c1ab8cd`。

## 启动 100K 服务

默认使用 GPU4–7 和端口1112，不会覆盖 GPU0–3/1111 的 8K 回滚服务：

```bash
cd /data/qwen38/deployment/qwen38-bi100
bash scripts/start_long_context_server.sh
```

等到 ready：

```bash
docker logs -f --tail=200 qwen38-bi100-longctx
curl -fsS http://127.0.0.1:1112/health
curl -fsS http://127.0.0.1:1112/v1/models | python3 -m json.tool
```

模型列表应显示：

```json
"max_model_len": 100000
```

## 真实长提示验收

脚本会把两个唯一识别码埋在提示词约 10% 和 90% 处，调用 `/tokenize` 贴近目标长度，再要求模型按顺序取回：

```bash
python3 scripts/test_long_context.py \
  --base-url http://QWEN_HOST:1112 \
  --target-input-tokens 96000
```

本次结果：

| 目标 | 实际 prompt tokens | 耗时 | 结果 |
|---:|---:|---:|---|
| 16K | 15,971 | 26.684s | 两个识别码正确 |
| 32K | 31,965 | 57.354s | 两个识别码正确 |
| 64K | 63,953 | 142.991s | 两个识别码正确 |
| 96K | 95,963 | 255.270s | 两个识别码正确 |

## 1M YaRN 探针

创建轻量 overlay；权重使用硬链接，只有 `config.json` 独立，因此不会重复占用约56GB：

```bash
python3 scripts/prepare_yarn_model.py \
  /data/qwen38/models/Qwen3.8-27B \
  /data/qwen38/models/Qwen3.8-27B-YaRN-1M \
  --factor 4.0 \
  --target-context 1000000
```

本次 TP4 探针成功加载 18 个权重分片和 YaRN 配置，但 profiling 结果为：

```text
# GPU blocks: 16371
Maximum concurrency for 1000000 tokens per request: 0.26x
maximum number of tokens that can be stored in KV cache: 261936
```

因此 vLLM 在 API ready 之前正确退出。提高 Codex 的 `model_context_window` 不会增加服务器 KV cache；强行绕过检查只会导致 OOM 或错误输出。

要在这类硬件上继续接近 1M，需要后端减少只用于 full-attention 层之外的 KV 分配，或采用经过验证的 pipeline parallel/KV 量化与更多显存。当前社区 dense adapter 没有这套实现，不能把它写进生产配置。

