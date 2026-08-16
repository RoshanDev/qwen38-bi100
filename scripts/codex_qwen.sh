#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
DATA_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}"
RUNTIME_DIR="${QWEN_CODEX_RUNTIME_DIR:-${DATA_ROOT}/codex-qwen38}"

if [[ ! -f "${RUNTIME_DIR}/config.toml" ]]; then
  echo "missing isolated Codex config: ${RUNTIME_DIR}/config.toml" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/qwen_bridge_ctl.sh" start >/dev/null
exec env CODEX_HOME="${RUNTIME_DIR}" codex "$@"
