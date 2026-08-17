# BI-V100 长上下文适配与实测

## 最终结论

- Qwen3.8-27B 官方原生上下文为262,144 tokens，可用静态 YaRN 扩到1M。
- 4×BI-V100 32GB、TP=4、CoreX/vLLM 0.6.3 上，400K/factor2 服务已完成398,971 prompt tokens 的双位置口令检索。
- 1M/factor4 服务已成功启动，KV容量1,047,760 tokens，并通过16K回归；未执行接近1M的完整提示。
- 400K冷请求耗时约44分35秒；容量可用不等于交互速度可用。

官方模型说明：<https://huggingface.co/Qwen/Qwen3.8-27B>

## 根因：64层被错误分配KV

模型有64层，但只有16层是 `full_attention`，其余48层是 `linear_attention`。当前 vLLM 0.6.3 的缓存计算读取顶层：

```python
layers = getattr(hf_config, "layers_block_type", ["attention"] * num_layers)
```

Qwen配置只在 `text_config.layer_types` 保存混合层布局，顶层字段不存在，vLLM便按64个attention层计算和创建KV cache；模型前向实际只使用16份。

旧探针数据恰好闭合：

```text
16371 blocks × 16 tokens = 261936 tokens
```

`prepare_yarn_model.py` 将 `full_attention` 映射成顶层 `attention`，保留 `linear_attention`。修复后动态调用返回16层，实机结果为：

| overlay | GPU blocks | KV tokens | 服务结果 |
|---|---:|---:|---|
| 400K/factor2 | 66,149（基础）/65,629（CHUNK候选） | 约1.05M | ready |
| 1M/factor4 | 65,485 | 1,047,760 | ready，1.05x concurrency |

## 构建镜像

先按固定 dense 提交 `d972854fb79f47aa6ab1b9a5a45d27ca99c8c5ab` 构建长上下文镜像：

```bash
COREX_BASE_IMAGE='YOUR_AUTHORIZED_COREX_3_2_3_LLM_IMAGE:v1.2.3' \
QWEN38_BASE_IMAGE='qwen38-bi100:corex3.2.3-text-0e899' \
ADAPTER_REPO_DIR='/data/qwen38/src/enginex-vllm-bi100-qwen36' \
BUILD_ROOT='/data/qwen38/build' \
bash scripts/build_long_context_image.sh
```

再构建本项目的 dense native 实验层：

```bash
BASE_IMAGE='qwen38-bi100:corex3.2.3-longctx-d972' \
OUTPUT_IMAGE='qwen38-bi100:corex3.2.3-dense-native-v1' \
bash scripts/build_dense_native_image.sh
```

实测 image ID：

```text
sha256:5ec386a3d4cb862eae915f2597d3ef945b0fd2d0eb7371edb3ea8be38214ce20
```

三个fastpath可独立控制，默认只有通过同轮A/B的CHUNK开启：

```text
QWEN38_LOOP1_NATIVE=0
QWEN38_CHUNK_PARALLEL=1
QWEN38_RMSNORM_NATIVE=0
```

## 创建overlay

权重使用硬链接；只有配置和manifest独立，不会重复占用约56GB：

```bash
python3 scripts/prepare_yarn_model.py \
  /data/qwen38/models/Qwen3.8-27B \
  /data/qwen38/models/Qwen3.8-27B-YaRN-400K-kvfix \
  --factor 2.0 \
  --target-context 400000

python3 scripts/prepare_yarn_model.py \
  /data/qwen38/models/Qwen3.8-27B \
  /data/qwen38/models/Qwen3.8-27B-YaRN-1M-kvfix \
  --factor 4.0 \
  --target-context 1000000
```

脚本会校验64层布局中恰有16个attention层，并确认源配置哈希在创建前后不变。

## 启动400K服务

默认配置使用GPU4–7、端口1112、TP4、4096 chunk：

```bash
cd /data/qwen38/deployment/qwen38-bi100
bash scripts/start_long_context_server.sh
```

检查：

```bash
docker logs -f --tail=200 qwen38-bi100-400k
curl -fsS http://127.0.0.1:1112/health
curl -fsS http://127.0.0.1:1112/v1/models | python3 -m json.tool
```

## 启动1M专用端点

1M不作为日常Codex端点。需要容量探针时，使用独立容器和端口：

```bash
CONTAINER_NAME=qwen38-bi100-1m \
MODEL_DIR=/data/qwen38/models/Qwen3.8-27B-YaRN-1M-kvfix \
PORT=1113 \
MAX_MODEL_LEN=1000000 \
QWEN38_CHUNK_PARALLEL=1 \
bash scripts/start_long_context_server.sh
```

预期启动日志：

```text
# GPU blocks: 65485
Maximum concurrency for 1000000 tokens per request: 1.05x
```

同一组GPU不能同时运行400K与1M服务；切换前先停止GPU4–7上的另一个容器。GPU0–3/1111的8K回滚服务可保持运行。

## 正确性与性能实测

测试把两个唯一识别码放在提示约10%和90%位置：

```bash
python3 scripts/test_long_context.py \
  --base-url http://QWEN_HOST:1112 \
  --target-input-tokens 399000 \
  --timeout-seconds 7200
```

| 服务/配置 | 实际 prompt | 耗时 | 结果 |
|---|---:|---:|---|
| 原100K路径 | 15,971 | 26.684s | 通过 |
| 原100K路径 | 31,965 | 57.354s | 通过 |
| 原100K路径 | 63,953 | 142.991s | 通过 |
| 原100K路径 | 95,963 | 255.270s | 通过 |
| 400K/CHUNK | 63,953 | 134.966s | 通过 |
| 400K/CHUNK | 271,965 | 1,350.743s | 通过 |
| 400K/CHUNK | 398,971 | 2,675.417s | 通过 |
| 1M/factor4 | 15,971 | 26.327s | 通过 |

同轮64K flags=0基线为146.886s，CHUNK为134.966s，改善8.12%。LOOP1两轮平均无收益；RMSNorm decode改善与噪声重叠；8192 chunk比4096慢且减少KV blocks。短生成仍约7.3–7.6 tok/s。

## 性能边界

- 400K和1M的主要问题已从KV容量转为full-attention冷prefill计算量。
- 当前adaptation没有MTP；单靠显存利用率、TP8或更大chunk无法把约7.5 tok/s提升到20–30 tok/s。
- CUDA Graph在社区同类CoreX+xFormers+TP4路径有OOM/挂死记录；custom all-reduce单并发无稳定收益，默认不启用。
- PREFIX_FLASH只覆盖有限前缀，且底层BI-V100路径有挂死历史；本项目没有把它放进默认镜像。
- 真正提高decode需要移植Qwen MTP/speculative worker或厂商提供的新vLLM/CoreX优化；这不是配置开关级改动。
