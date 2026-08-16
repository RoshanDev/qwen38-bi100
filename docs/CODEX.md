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
curl --noproxy '*' -fsS http://QWEN_HOST:1112/health
```

安装隔离配置、用户级 systemd 服务和启动命令：

```bash
bash scripts/install_codex_integration.sh http://QWEN_HOST:1112/v1
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
codex-qwen38 exec --strict-config --ephemeral --skip-git-repo-check --ignore-rules -C /tmp/qwen38-codex-test '只回答 CODEX_QWEN_OK，不要调用工具。'
```

这是最不容易复制错的单行版本。提示词已经放在单引号中，`CODEX_QWEN_OK` 的下划线不需要写成 `\_`。

如果希望换行书写，必须只使用一个反斜杠，而且反斜杠必须是该行最后一个字符：

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

不要把命令写成 `exec \\ --strict-config` 或 `exec \ --strict-config`：前者会把 `\` 作为参数，后者会把反斜杠后的空格合并进下一个参数。

预期最后一行是：

```text
CODEX_QWEN_OK
```

再测试真实工具调用：

```bash
codex-qwen38 exec --strict-config --ephemeral --skip-git-repo-check --ignore-rules -C /tmp/qwen38-codex-test '必须调用 exec_command 工具执行命令 printf CODEX_TOOL_OK，然后只回答该命令的输出。'
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

可以用与此前中断场景相同的短提示验证：

```text
哈喽你好
```

本次实测返回正常中文问候，没有 `Conversation interrupted`。

正常的 OpenAI Codex 仍使用原命令：

```bash
codex
```

## 100K 上下文预算

Codex 的系统提示、工具 schema 和项目说明在本机首轮约占 8K tokens，因此 8K 服务没有实际交互余量。当前配置为：

- Codex `model_context_window=100000`；
- 90,000 tokens 触发 Codex 历史压缩；
- bridge 输入安全线为 94,000 tokens；
- 单次输出上限 4,096 tokens，安全 margin 256；
- 单条工具输出最多保留 16,000 字符，单 turn 最多 32 次工具调用；
- 不再压缩 Codex 内建系统提示；
- 调用上游 `/tokenize` 计算真实输入大小，并动态缩小输出预算；
- 超过安全线时先截短旧工具输出，再由 Codex 自动压缩历史。

本次普通回复和工具闭环分别实际使用 8,319 与 16,787 tokens，均通过。`Model metadata ... not found` 表示 Codex 没有内置此自定义模型的产品元数据；skills 描述缩短警告表示所有 skill 仍可见但描述更短。两者不影响已设置的 100K context；真正失败会显示 HTTP 状态或 `ERROR`。

模型服务最高已用 95,963 prompt tokens 做双位置口令检索。完整记录见 [长上下文适配与实测](LONG_CONTEXT.md)。

## 明确指定隔离配置

如果不使用启动脚本，可在单条命令中指定专用配置目录：

```bash
CODEX_HOME="$HOME/.local/share/codex-qwen38" codex exec '只回答 OK'
```

Codex 也支持命名 profile，但 custom provider 配置不支持放在项目 `.codex/config.toml` 中。这里选择独立 `CODEX_HOME`，可以把配置、状态和认证边界一起隔离。

配置字段与优先级可对照 [OpenAI Codex configuration reference](https://developers.openai.com/codex/config-reference)。本仓库已实测 Codex CLI `0.147.0`。

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
