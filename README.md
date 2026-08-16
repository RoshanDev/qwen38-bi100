# Qwen3.8-27B on Iluvatar BI-V100

公开仓库：<https://github.com/RoshanDev/qwen38-bi100>

## 已验证状态

2026-08-16 已在一台 8×BI-V100 32GB 服务器上实测部署成功，服务当前保持运行。

| 项目 | 值 |
|---|---|
| API | `http://QWEN_HOST:1111/v1` |
| 服务容器 | `qwen38-bi100-server` |
| 派生镜像 | `qwen38-bi100:corex3.2.3-text-0e899` |
| 模型目录 | `/data/qwen38/models/Qwen3.8-27B` |
| GPU | 0–3，TP=4 |
| 上下文 | 8192 |
| 运行方式 | FP16、eager、`max-num-seqs=1` |
| 稳态显存 | 约 24.9–25.1GB/卡 |
| 512-token 端到端吞吐 | 7.985 tok/s |

模型固定在官方 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。18 个权重分片、1199 个索引键和总字节均已离线校验。

## 调用示例

如果客户端设置了全局代理，访问内网地址时需要绕过代理：

```bash
curl --noproxy '*' -sS \
  http://QWEN_HOST:1111/v1/chat/completions \
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

重新执行正确性测试和基准：

```bash
python3 /data/qwen38/scripts/smoke_api.py --wait-seconds 60
python3 /data/qwen38/scripts/benchmark_api.py --max-tokens 512
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

## 已知边界

- 当前适配仅验证文本推理，不支持图片输入。
- 未启用 MTP 推测解码、FP8、chunked prefill 或 CUDA graph。
- 第一轮只验证到 8K 上下文；官方 262K 上限不能据此视为已支持。
- 服务使用 GPU 0–3；GPU 4–7 保持空闲。
- CoreX 基础镜像未随仓库分发，也没有发布到 Docker Hub；当前没有找到足以证明该私有运行时可公开再分发的授权文本。

## Codex CLI

仓库提供 Responses → Chat Completions 本地 bridge 和完全隔离的 Codex 配置。它不会覆盖 `~/.codex/config.toml`；OpenAI 官方 Codex 继续使用 `codex`，Qwen 入口使用 `codex-qwen38`。

```bash
bash scripts/install_codex_integration.sh http://QWEN_HOST:1111/v1
codex-qwen38 exec --ephemeral --skip-git-repo-check '只回答 CODEX_QWEN_OK'
```

安装、普通回复测试、真实工具调用测试及卸载方法见 [Codex CLI 接入与手动验收](docs/CODEX.md)。

## 来源

- [Qwen3.8-27B 官方模型](https://huggingface.co/Qwen/Qwen3.8-27B)
- [BI-V100 社区适配提交](https://dev.modelhub.org.cn/icer/qwen36_01/commit/0e89906481e9a6cb2475925adb41517676e12903)
- [社区四卡运行记录](https://dev.modelhub.org.cn/icer/qwen36_01/src/branch/main/worklogs/2026-07-13-initial-run.md)
- [用户提供的天数知识库参考页](https://ixkb.iluvatar.com.cn:9443/webdoc/view/Pub8a16948a9a4cb023019cbca37f861590.html)
