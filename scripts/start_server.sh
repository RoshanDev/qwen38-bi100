#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-qwen38-bi100:corex3.2.3-text-0e899}"
MODEL_DIR="${MODEL_DIR:-/data/qwen38/models/Qwen3.8-27B}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-bi100-server}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PORT="${PORT:-1111}"

if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "model is incomplete: ${MODEL_DIR}" >&2
  exit 1
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
  --entrypoint python3 \
  "${IMAGE}" \
  -m vllm.entrypoints.openai.api_server \
  --model /model \
  --port "${PORT}" \
  --served-model-name Qwen3.8-27B \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --enforce-eager \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90
