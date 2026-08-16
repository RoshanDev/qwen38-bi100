#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-qwen38-bi100:corex3.2.3-longctx-d972}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-qwen38-bi100:corex3.2.3-dense-native-v1}"

docker image inspect "${BASE_IMAGE}" >/dev/null
docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "${OUTPUT_IMAGE}" \
  --file "${PROJECT_ROOT}/docker/Dockerfile.dense-native" \
  "${PROJECT_ROOT}"

docker image inspect "${OUTPUT_IMAGE}" \
  --format '{{.Id}} {{.Size}}'
