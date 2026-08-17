#!/usr/bin/env bash
# Build isolated MTP image, A/B on :1113, restore 8K. Never touch 400K.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/data/qwen38/logs/mtp-k1-exp
STATUS="${LOG}/STATUS.txt"
mkdir -p "${LOG}"
say() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "${STATUS}"; }

need_400k() {
  local code
  code="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:1112/health || true)"
  [[ "${code}" == "200" ]] || { say "ABORT 400K=${code}"; exit 20; }
}

wait_http() {
  local url="$1" timeout="${2:-900}" code=""
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    code="$(curl -sS -m 4 -o /dev/null -w '%{http_code}' "${url}" || true)"
    [[ "${code}" == "200" ]] && return 0
    sleep 5
  done
  return 1
}

restore_8k() {
  docker rm -f qwen38-bi100-mtp-k1-exp >/dev/null 2>&1 || true
  docker start qwen38-bi100-server >/dev/null || true
  if wait_http http://127.0.0.1:1111/health 600; then
    say "8K restored"
  else
    say "FAIL 8K restore"
  fi
}
trap 'say "trap"; restore_8k; need_400k || true' EXIT

need_400k
say "build image"
bash "${ROOT}/scripts/build_mtp_k1_image.sh" | tee -a "${LOG}/build.log"

need_400k
say "stop 8K keep container"
docker inspect qwen38-bi100-server > "${LOG}/qwen38-bi100-server.inspect.before-exp.json"
docker stop qwen38-bi100-server
sleep 3
need_400k

for mode in 0 1; do
  need_400k
  say "start exp MTP=${mode}"
  QWEN38_MTP_K1="${mode}" bash "${ROOT}/scripts/start_mtp_k1_exp.sh"
  if ! wait_http http://127.0.0.1:1113/health 900; then
    docker logs qwen38-bi100-mtp-k1-exp > "${LOG}/exp-mtp${mode}.container.log" 2>&1 || true
    docker inspect qwen38-bi100-mtp-k1-exp > "${LOG}/exp-mtp${mode}.inspect.json" 2>/dev/null || true
    say "FAIL exp health MTP=${mode}"
    exit 21
  fi
  python3 "${ROOT}/scripts/run_mtp_k1_eval.py" \
    --base-url http://127.0.0.1:1113 \
    --label "mtp${mode}" \
    --out-dir "${LOG}" \
    --max-tokens 64 256 \
    --prompts zh_explain go_code \
    --warmup 1 \
    --runs 2 \
    | tee -a "${LOG}/eval-mtp${mode}.stdout"
  docker rm -f qwen38-bi100-mtp-k1-exp >/dev/null
  sleep 3
done

need_400k
say "restore 8K"
restore_8k
need_400k
say "done"
exit 0
