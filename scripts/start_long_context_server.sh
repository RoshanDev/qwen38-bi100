#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-qwen38-bi100:corex3.2.3-longctx-d972}"
MODEL_DIR="${MODEL_DIR:-/data/qwen38/models/Qwen3.8-27B}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-bi100-longctx}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
PORT="${PORT:-1112}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-100000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"

if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "model is incomplete: ${MODEL_DIR}" >&2
  exit 1
fi
if ! [[ "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_MODEL_LEN must be a positive integer" >&2
  exit 2
fi
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "container already exists: ${CONTAINER_NAME}" >&2
  exit 1
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network=host \
  --ipc=host \
  --privileged \
  -v /usr/src:/usr/src:ro \
  -v /lib/modules:/lib/modules:ro \
  -v /dev:/dev \
  -v "${MODEL_DIR}:/model:ro" \
  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --entrypoint python3 \
  "${IMAGE}" \
  -m vllm.entrypoints.openai.api_server \
  --model /model \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --enable-chunked-prefill \
  --disable-log-requests \
  --disable-frontend-multiprocessing \
  --enforce-eager \
  --trust-remote-code \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
