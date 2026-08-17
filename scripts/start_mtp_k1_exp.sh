#!/usr/bin/env bash
# Independent MTP experiment on GPU0-3 / :1113. Never touches 400K.
set -euo pipefail
IMAGE="${IMAGE:-qwen38-bi100:corex3.2.3-mtp-k1-exp}"
MODEL_DIR="${MODEL_DIR:-/data/qwen38/models/Qwen3.8-27B}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-bi100-mtp-k1-exp}"
PORT="${PORT:-1113}"
QWEN38_MTP_K1="${QWEN38_MTP_K1:-0}"
LOG_DIR="${LOG_DIR:-/data/qwen38/logs/mtp-k1-exp}"

if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "model is incomplete: ${MODEL_DIR}" >&2
  exit 1
fi
if docker inspect --format '{{.State.Status}}' qwen38-bi100-400k 2>/dev/null | grep -qx running; then
  :
else
  echo "refusing: 400K is not running" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${CONTAINER_NAME}" \
  --network=host \
  --ipc=host \
  --privileged \
  -v /usr/src:/usr/src:ro \
  -v /lib/modules:/lib/modules:ro \
  -v /dev:/dev \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${LOG_DIR}:/logs/mtp-k1-exp" \
  -v /data/qwen38/src/mtp-k1-exp/scripts/mtp_k1:/opt/mtp_k1:ro \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e QWEN38_MTP_K1="${QWEN38_MTP_K1}" \
  -e MTP_MODEL=/model \
  -e MTP_LOG_DIR=/logs/mtp-k1-exp \
  --entrypoint python3 \
  "${IMAGE}" \
  -m vllm.entrypoints.openai.api_server \
  --model /model \
  --port "${PORT}" \
  --served-model-name Qwen3.8-27B \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --enforce-eager \
  --disable-frontend-multiprocessing \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.80
echo "started ${CONTAINER_NAME} port=${PORT} MTP=${QWEN38_MTP_K1}"
