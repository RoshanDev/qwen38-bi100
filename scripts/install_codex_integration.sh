#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}"
CONFIG_ROOT="${XDG_CONFIG_HOME:-${HOME}/.config}"
BIN_ROOT="${HOME}/.local/bin"
RUNTIME_DIR="${QWEN_CODEX_RUNTIME_DIR:-${DATA_ROOT}/codex-qwen38}"
SERVICE_FILE="${CONFIG_ROOT}/systemd/user/qwen38-codex-bridge.service"
UPSTREAM_URL="${1:-${QWEN_UPSTREAM_BASE_URL:-}}"

if [[ -z "${UPSTREAM_URL}" ]]; then
  echo "usage: $0 http://QWEN_HOST:1112/v1" >&2
  exit 2
fi

mkdir -p "${RUNTIME_DIR}" "$(dirname "${SERVICE_FILE}")" "${BIN_ROOT}"

if [[ ! -f "${RUNTIME_DIR}/config.toml" ]]; then
  install -m 600 "${PROJECT_ROOT}/codex/config.example.toml" "${RUNTIME_DIR}/config.toml"
else
  echo "preserving existing isolated config: ${RUNTIME_DIR}/config.toml"
fi

if [[ ! -f "${RUNTIME_DIR}/bridge.env" ]]; then
  umask 077
  {
    printf 'QWEN_UPSTREAM_BASE_URL=%s\n' "${UPSTREAM_URL}"
    printf 'QWEN_UPSTREAM_TIMEOUT_SECONDS=3600\n'
    printf 'QWEN_UPSTREAM_USE_SYSTEM_PROXY=0\n'
    printf 'QWEN_MAX_CONTEXT_TOKENS=400000\n'
    printf 'QWEN_MAX_INPUT_TOKENS=390000\n'
    printf 'QWEN_MAX_TOOL_OUTPUT_CHARS=16000\n'
    printf 'QWEN_MAX_TOOL_CALLS_PER_TURN=32\n'
    printf 'QWEN_MAX_OUTPUT_TOKENS=8192\n'
    printf 'QWEN_TOKEN_SAFETY_MARGIN=256\n'
    printf 'QWEN_FALLBACK_MAX_OUTPUT_TOKENS=8192\n'
    printf 'QWEN_COMPACT_CODEX_INSTRUCTIONS=0\n'
    printf 'QWEN_BRIDGE_HOST=127.0.0.1\n'
    printf 'QWEN_BRIDGE_PORT=8348\n'
    printf 'QWEN_VISION_BASE_URL=\n'
    printf 'QWEN_VISION_MODEL=Qwen3.8-27B\n'
    printf 'QWEN_VISION_API_KEY=\n'
    printf 'QWEN_VISION_TIMEOUT_SECONDS=60\n'
    printf 'QWEN_VISION_OCR_LANGS=chi_sim+eng\n'
  } >"${RUNTIME_DIR}/bridge.env"
else
  echo "preserving existing bridge environment: ${RUNTIME_DIR}/bridge.env"
fi

install -m 644 "${PROJECT_ROOT}/codex/model-catalog.json" "${RUNTIME_DIR}/model-catalog.json"
CATALOG_PATH="${RUNTIME_DIR}/model-catalog.json" python3 - <<'PY'
from pathlib import Path
import os
import re

runtime = Path(os.environ["CATALOG_PATH"]).parent
catalog = runtime / "model-catalog.json"
config = runtime / "config.toml"
line = f'model_catalog_json = "{catalog}"'
text = config.read_text()
if re.search(r"^model_catalog_json\s*=", text, re.M):
    text = re.sub(r"^model_catalog_json\s*=.*$", line, text, count=1, flags=re.M)
elif re.search(r"^model_provider\s*=", text, re.M):
    text = re.sub(
        r"^(model_provider\s*=.*)$",
        r"\1\n" + line,
        text,
        count=1,
        flags=re.M,
    )
else:
    text = line + "\n" + text
config.write_text(text)

env_file = runtime / "bridge.env"
existing = env_file.read_text() if env_file.exists() else ""
defaults = {
    "QWEN_VISION_BASE_URL": "",
    "QWEN_VISION_MODEL": "Qwen3.8-27B",
    "QWEN_VISION_API_KEY": "",
    "QWEN_VISION_TIMEOUT_SECONDS": "60",
    "QWEN_VISION_OCR_LANGS": "chi_sim+eng",
}
missing = [
    f"{key}={value}\n"
    for key, value in defaults.items()
    if not re.search(rf"^{re.escape(key)}=", existing, re.M)
]
if missing:
    with env_file.open("a") as handle:
        handle.writelines(missing)
PY

escaped_project="${PROJECT_ROOT//|/\\|}"
escaped_runtime="${RUNTIME_DIR//|/\\|}"
sed \
  -e "s|@PROJECT_ROOT@|${escaped_project}|g" \
  -e "s|@RUNTIME_DIR@|${escaped_runtime}|g" \
  "${PROJECT_ROOT}/systemd/qwen38-codex-bridge.service.example" >"${SERVICE_FILE}"

ln -sfn "${PROJECT_ROOT}/scripts/codex_qwen.sh" "${BIN_ROOT}/codex-qwen38"
systemctl --user daemon-reload
systemctl --user enable --now qwen38-codex-bridge.service

echo "installed isolated Codex home: ${RUNTIME_DIR}"
echo "installed launcher: ${BIN_ROOT}/codex-qwen38"
echo "bridge health: http://127.0.0.1:8348/healthz"
