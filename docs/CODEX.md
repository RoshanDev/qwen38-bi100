# Codex CLI 接入与手动验收

这套接入不会修改 `~/.codex/config.toml`，也不会复用或覆盖 OpenAI 官方登录。它使用单独的 `CODEX_HOME`：

```text
~/.local/share/codex-qwen38/
├── config.toml
└── bridge.env
```

原因是当前 Codex custom provider 使用 Responses API，而已部署的 CoreX vLLM 只提供 Chat Completions。仓库中的本地 bridge 完成两种协议的转换，并把 Qwen XML 工具调用转换回 Codex 能执行的 function call。

## 安装

先确认直接 API 可访问，把 `QWEN_HOST` 换成推理服务器地址：

```bash
curl --noproxy '*' -fsS http://QWEN_HOST:1111/health
```

安装隔离配置、用户级 systemd 服务和启动命令：

```bash
bash scripts/install_codex_integration.sh http://QWEN_HOST:1111/v1
```

脚本只写入以下独立位置：

```text
~/.local/share/codex-qwen38/
~/.config/systemd/user/qwen38-codex-bridge.service
~/.local/bin/codex-qwen38
```

如果隔离配置或 bridge 环境文件已经存在，安装脚本会保留原文件。确认服务：

```bash
systemctl --user status qwen38-codex-bridge.service --no-pager
curl --noproxy '*' -fsS http://127.0.0.1:8348/healthz
```

## 手动测试

测试普通回答：

```bash
mkdir -p /tmp/qwen38-codex-test
codex-qwen38 exec \
  --strict-config \
  --ephemeral \
  --skip-git-repo-check \
  --ignore-rules \
  -C /tmp/qwen38-codex-test \
  '只回答 CODEX_QWEN_OK，不要调用工具。'
```

预期最后一行是：

```text
CODEX_QWEN_OK
```

再测试真实工具调用：

```bash
codex-qwen38 exec \
  --strict-config \
  --ephemeral \
  --skip-git-repo-check \
  --ignore-rules \
  -C /tmp/qwen38-codex-test \
  '必须调用 exec_command 工具执行命令 printf CODEX_TOOL_OK，然后只回答该命令的输出。'
```

输出中应同时出现工具执行记录和最终答案：

```text
exec
/usr/bin/zsh -lc 'printf CODEX_TOOL_OK'
CODEX_TOOL_OK
```

交互模式直接运行：

```bash
codex-qwen38
```

正常的 OpenAI Codex 仍使用原命令：

```bash
codex
```

## 明确指定隔离配置

如果不使用启动脚本，可在单条命令中指定专用配置目录：

```bash
CODEX_HOME="$HOME/.local/share/codex-qwen38" codex exec '只回答 OK'
```

Codex 也支持命名 profile，但 custom provider 配置不支持放在项目 `.codex/config.toml` 中。这里选择独立 `CODEX_HOME`，可以把配置、状态和认证边界一起隔离。

## 日志与恢复

```bash
journalctl --user -u qwen38-codex-bridge.service -n 100 --no-pager
systemctl --user restart qwen38-codex-bridge.service
```

停止隔离接入不会影响官方 Codex：

```bash
systemctl --user disable --now qwen38-codex-bridge.service
```

如需彻底移除，可在确认目录内容无须保留后，手动删除上面列出的三个隔离目标。推理服务器和 `~/.codex/config.toml` 均不会受到影响。
