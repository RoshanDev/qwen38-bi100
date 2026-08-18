#!/usr/bin/env bash
# Gate 5: dump target traces on GPU0-3, restore 8K, CPU-replay MTP KV.
# Never touches 400K / GPU4-7. Never docker rm qwen38-bi100-server.
set -euo pipefail

LOGDIR=/data/qwen38/logs/mtp-gate5-kv
SCRIPT_DIR=/data/qwen38/scripts
STATUS="${LOGDIR}/STATUS.txt"
PROD8K=qwen38-bi100-server
DUMP_NAME=qwen38-mtp-gate5-dump
REPLAY_NAME=qwen38-mtp-gate5-replay
IMAGE=qwen38-bi100:corex3.2.3-text-0e899
STOPPED_8K=0
mkdir -p "${LOGDIR}"

say() {
  local line
  line="$(date -Iseconds) $*"
  printf '%s\n' "${line}" | tee -a "${STATUS}"
}

need_400k() {
  local code
  code="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:1112/health || true)"
  if [[ "${code}" != "200" ]]; then
    say "ABORT 400K health=${code}"
    exit 20
  fi
}

wait_http() {
  local url="$1" timeout="${2:-600}" code=""
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    code="$(curl -sS -m 4 -o /dev/null -w '%{http_code}' "${url}" || true)"
    if [[ "${code}" == "200" ]]; then
      return 0
    fi
    sleep 5
  done
  return 1
}

wait_gpu_free() {
  local i
  for i in $(seq 1 36); do
    if ! docker inspect -f '{{.State.Running}}' "${PROD8K}" 2>/dev/null | grep -qx true; then
      sleep 5
      return 0
    fi
    sleep 5
  done
}

restore_8k() {
  docker rm -f "${DUMP_NAME}" >/dev/null 2>&1 || true
  if [[ "${STOPPED_8K}" == "1" ]]; then
    say "restore 8K container"
    docker start "${PROD8K}" >/dev/null || true
    if wait_http http://127.0.0.1:1111/health 600; then
      say "8K restored health=200"
    else
      say "FAIL 8K restore health timeout"
    fi
  fi
}

cleanup() {
  local rc=$?
  say "gate5 cleanup rc=${rc}"
  restore_8k
  docker rm -f "${DUMP_NAME}" >/dev/null 2>&1 || true
  need_400k || true
  say "gate5 cleanup done"
}
trap cleanup EXIT

say "mtp gate5 kv start"
need_400k
if ! wait_http http://127.0.0.1:1111/health 30; then
  say "WARN 8K not healthy before stop; still attempting dump"
fi

say "stop ${PROD8K} (keep container)"
STOPPED_8K=1
docker stop "${PROD8K}"
wait_gpu_free
need_400k

say "run ${DUMP_NAME}"
docker rm -f "${DUMP_NAME}" >/dev/null 2>&1 || true
docker run --name "${DUMP_NAME}" \
  --ipc=host \
  --privileged \
  --network=none \
  -v /usr/src:/usr/src:ro \
  -v /lib/modules:/lib/modules:ro \
  -v /dev:/dev \
  -v /data/qwen38/models/Qwen3.8-27B:/model:ro \
  -v "${SCRIPT_DIR}:/scripts:ro" \
  -v "${LOGDIR}:/logs" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e QWEN38_MTP_K1=0 \
  -e MTP_GATE5_DUMP=/logs/raw \
  --entrypoint python3 \
  "${IMAGE}" \
  /scripts/mtp_gate5_dump.py \
  --model /model \
  --out /logs/mtp-gate5-dump.json \
  --max-tokens 160 \
  > "${LOGDIR}/mtp-gate5-dump.stdout" 2>&1

say "dump finished"
restore_8k
STOPPED_8K=0
need_400k

say "cpu replay while 8K is up"
docker rm -f "${REPLAY_NAME}" >/dev/null 2>&1 || true
docker run --name "${REPLAY_NAME}" \
  --network=none \
  -v /data/qwen38/models/Qwen3.8-27B:/model:ro \
  -v "${SCRIPT_DIR}:/scripts:ro" \
  -v "${LOGDIR}:/logs" \
  -e CUDA_VISIBLE_DEVICES= \
  --entrypoint python3 \
  "${IMAGE}" \
  /scripts/mtp_gate5_kv_replay.py \
  --model /model \
  --dump /logs/mtp-gate5-dump.pt \
  --out /logs/mtp-gate5-replay.json \
  > "${LOGDIR}/mtp-gate5-replay.stdout" 2>&1

docker rm -f "${REPLAY_NAME}" >/dev/null 2>&1 || true
say "replay finished"
need_400k
if ! wait_http http://127.0.0.1:1111/health 30; then
  say "FAIL 8K unhealthy after replay"
  exit 21
fi
say "MTP_GATE5_KV_DONE"
