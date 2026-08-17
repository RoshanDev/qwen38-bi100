#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-qwen38-bi100:corex3.2.3-text-0e899}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-qwen38-bi100:corex3.2.3-mtp-k1-exp}"
docker image inspect "${BASE_IMAGE}" >/dev/null
docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "${OUTPUT_IMAGE}" \
  --file "${PROJECT_ROOT}/docker/Dockerfile.mtp-k1" \
  "${PROJECT_ROOT}"
docker image inspect "${OUTPUT_IMAGE}" --format '{{.Id}} {{.Size}}'
