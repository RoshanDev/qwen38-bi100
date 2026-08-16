#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}"
STATE_ROOT="${XDG_STATE_HOME:-${HOME}/.local/state}"
RUNTIME_DIR="${QWEN_CODEX_RUNTIME_DIR:-${DATA_ROOT}/codex-qwen38}"
STATE_DIR="${STATE_ROOT}/codex-qwen38"
ENV_FILE="${RUNTIME_DIR}/bridge.env"
PID_FILE="${STATE_DIR}/bridge.pid"
LOG_FILE="${STATE_DIR}/bridge.log"
BRIDGE_SCRIPT="${PROJECT_ROOT}/bridge/responses_to_chat.py"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing bridge environment: ${ENV_FILE}" >&2
  exit 1
fi
if [[ ! -f "${BRIDGE_SCRIPT}" ]]; then
  echo "missing bridge program: ${BRIDGE_SCRIPT}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

BRIDGE_HOST="${QWEN_BRIDGE_HOST:-127.0.0.1}"
BRIDGE_PORT="${QWEN_BRIDGE_PORT:-8348}"
HEALTH_URL="http://${BRIDGE_HOST}:${BRIDGE_PORT}/healthz"

health_check() {
  curl --noproxy '*' -fsS --max-time 3 "${HEALTH_URL}" >/dev/null 2>&1
}

start_bridge() {
  local pid
  if health_check; then
    echo "bridge already healthy: ${HEALTH_URL}"
    return 0
  fi

  mkdir -p "${STATE_DIR}"
  nohup env \
    QWEN_UPSTREAM_BASE_URL="${QWEN_UPSTREAM_BASE_URL}" \
    QWEN_UPSTREAM_API_KEY="${QWEN_UPSTREAM_API_KEY:-}" \
    QWEN_UPSTREAM_TIMEOUT_SECONDS="${QWEN_UPSTREAM_TIMEOUT_SECONDS:-3600}" \
    QWEN_UPSTREAM_USE_SYSTEM_PROXY="${QWEN_UPSTREAM_USE_SYSTEM_PROXY:-0}" \
    QWEN_MAX_CONTEXT_TOKENS="${QWEN_MAX_CONTEXT_TOKENS:-400000}" \
    QWEN_MAX_INPUT_TOKENS="${QWEN_MAX_INPUT_TOKENS:-390000}" \
    QWEN_MAX_TOOL_OUTPUT_CHARS="${QWEN_MAX_TOOL_OUTPUT_CHARS:-16000}" \
    QWEN_MAX_TOOL_CALLS_PER_TURN="${QWEN_MAX_TOOL_CALLS_PER_TURN:-32}" \
    QWEN_MAX_OUTPUT_TOKENS="${QWEN_MAX_OUTPUT_TOKENS:-8192}" \
    QWEN_TOKEN_SAFETY_MARGIN="${QWEN_TOKEN_SAFETY_MARGIN:-256}" \
    QWEN_FALLBACK_MAX_OUTPUT_TOKENS="${QWEN_FALLBACK_MAX_OUTPUT_TOKENS:-8192}" \
    QWEN_COMPACT_CODEX_INSTRUCTIONS="${QWEN_COMPACT_CODEX_INSTRUCTIONS:-0}" \
    QWEN_BRIDGE_HOST="${BRIDGE_HOST}" \
    QWEN_BRIDGE_PORT="${BRIDGE_PORT}" \
    python3 "${BRIDGE_SCRIPT}" \
    --host "${BRIDGE_HOST}" \
    --port "${BRIDGE_PORT}" \
    >"${LOG_FILE}" 2>&1 </dev/null &
  pid=$!
  echo "${pid}" >"${PID_FILE}"

  for _ in {1..20}; do
    if health_check; then
      echo "bridge started: ${HEALTH_URL} (pid=${pid})"
      return 0
    fi
    sleep 0.25
  done

  echo "bridge failed to start; log: ${LOG_FILE}" >&2
  tail -40 "${LOG_FILE}" >&2 || true
  exit 1
}

stop_bridge() {
  local pid
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "bridge pid file does not exist: ${PID_FILE}"
    return 0
  fi
  pid="$(tr -d '[:space:]' <"${PID_FILE}")"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    echo "invalid bridge pid file: ${PID_FILE}" >&2
    exit 1
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    for _ in {1..20}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
  fi
  rm -f "${PID_FILE}"
  echo "bridge stopped"
}

status_bridge() {
  echo "runtime_dir=${RUNTIME_DIR}"
  echo "health_url=${HEALTH_URL}"
  echo "log_file=${LOG_FILE}"
  if health_check; then
    echo "status=healthy"
    return 0
  fi
  echo "status=down"
  return 1
}

case "${1:-status}" in
  start)
    start_bridge
    ;;
  stop)
    stop_bridge
    ;;
  restart)
    stop_bridge
    start_bridge
    ;;
  status)
    status_bridge
    ;;
  logs)
    tail -f "${LOG_FILE}"
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
