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
  } >"${RUNTIME_DIR}/bridge.env"
else
  echo "preserving existing bridge environment: ${RUNTIME_DIR}/bridge.env"
fi

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
