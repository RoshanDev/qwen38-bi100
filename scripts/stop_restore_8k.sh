#!/usr/bin/env bash
# Stop MTP experiment container and restore 8K. Never rm 8K. Never touch 400K.
set -euo pipefail
EXP="${EXP_CONTAINER:-qwen38-bi100-mtp-k1-exp}"
PROD8K="${PROD8K:-qwen38-bi100-server}"

need_400k() {
  local code
  code="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:1112/health || true)"
  if [[ "${code}" != "200" ]]; then
    echo "400K health=${code}" >&2
    return 1
  fi
}

need_400k
docker rm -f "${EXP}" >/dev/null 2>&1 || true
docker start "${PROD8K}" >/dev/null || true
for _ in $(seq 1 120); do
  if curl -sS -m 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:1111/health | grep -qx 200; then
    echo "8K restored health=200"
    need_400k
    exit 0
  fi
  sleep 5
done
echo "8K restore timeout" >&2
exit 1
