# 公共发布边界

## GitHub

可以公开。本仓库只包含原创编排脚本、文档和一个针对固定社区 commit 的小型补丁；不会复制模型权重、CoreX 文件或完整社区 adapter 源码。

## Docker Hub

本次不推送公开镜像。派生镜像包含来自私有/授权渠道的 CoreX 基础层；当前没有取得允许公开再分发这些二进制层的明确许可证文本。用户对自有环境的公开授权不能替代第三方权利人的许可。

可公开发布的是 Dockerfile、构建脚本、精确 commit 和验证结果。具备合法基础镜像的用户可在本地复现。

## Hugging Face

Qwen3.8-27B 官方权重使用 Apache-2.0；本次没有修改任何 tensor，只把 `architectures` 切换为文本 adapter，并可选生成一个 YaRN `config.json` overlay。重新上传约56GB相同权重没有技术收益，也容易让用户误以为这是新的训练/量化版本，因此本次不创建重复模型仓库。

需要 1M 配置时，使用 [prepare_yarn_model.py](../scripts/prepare_yarn_model.py) 从官方固定 revision 本地生成即可。若将来确需发布衍生模型，应保留 Apache-2.0 LICENSE/NOTICE、清楚标注修改，并先解决推理 adapter/CoreX 运行时的独立分发许可。

