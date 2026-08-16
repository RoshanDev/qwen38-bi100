# 备份与恢复

## 本次备份范围

应保存以下内容：

- 模型目录、官方配置备份和 Hugging Face revision 信息；
- 社区源码仓库及固定 commit；
- 构建上下文、部署脚本和运行日志；
- CoreX 基础镜像归档；
- 可直接 `docker load` 的派生镜像归档；
- `docker inspect`、镜像 inspect、GPU/OS/Docker 版本；
- WSL 隔离的 Codex `config.toml`、`bridge.env`、user systemd unit 和本仓库；
- YaRN overlay 的独立配置与生成脚本。

不要把 Codex history、SQLite、installation ID 或官方 `~/.codex` 登录状态放进共享备份。

## 不占 WSL 磁盘的主目录复制

下面的 WSL 只转发字节，不保存中间文件：

```bash
ssh root@QWEN_HOST \
  'tar -C /data -cf - qwen38' \
| ssh BACKUP_USER@BACKUP_HOST \
  'mkdir -p /home/BACKUP_USER/backup/qwen38-bi100-DATE/host &&
   tar -C /home/BACKUP_USER/backup/qwen38-bi100-DATE/host -xf -'
```

模型至少核验分片数量和三个关键哈希：

```bash
find models/Qwen3.8-27B -maxdepth 1 -name 'model-*.safetensors' | wc -l
sha256sum \
  models/Qwen3.8-27B/config.json \
  models/Qwen3.8-27B/config.json.qwen38-official \
  models/Qwen3.8-27B/model.safetensors.index.json
```

## 镜像归档

将当前派生镜像直接压缩到备份机：

```bash
ssh root@QWEN_HOST \
  'docker save qwen38-bi100:corex3.2.3-longctx-d972 | gzip -1' \
| ssh BACKUP_USER@BACKUP_HOST \
  'cat > /home/BACKUP_USER/backup/qwen38-bi100-DATE/images/qwen38-longctx.tar.gz'
```

目标端必须执行：

```bash
gzip -t qwen38-longctx.tar.gz
sha256sum qwen38-longctx.tar.gz > qwen38-longctx.tar.gz.sha256
```

恢复：

```bash
sha256sum -c qwen38-longctx.tar.gz.sha256
gzip -dc qwen38-longctx.tar.gz | docker load
```

## WSL Codex 配置

只保存这些路径：

```text
~/Developer/Qwen3.8-27B/
~/.local/share/codex-qwen38/config.toml
~/.local/share/codex-qwen38/bridge.env
~/.config/systemd/user/qwen38-codex-bridge.service
~/.local/bin/codex-qwen38
```

恢复后：

```bash
systemctl --user daemon-reload
systemctl --user enable --now qwen38-codex-bridge.service
curl --noproxy '*' -fsS http://127.0.0.1:8348/healthz
```

