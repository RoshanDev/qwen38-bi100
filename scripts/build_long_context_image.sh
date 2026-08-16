#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ADAPTER_REPO_DIR="${ADAPTER_REPO_DIR:-/data/qwen38/src/enginex-vllm-bi100-qwen36}"
ADAPTER_COMMIT="${ADAPTER_COMMIT:-d972854fb79f47aa6ab1b9a5a45d27ca99c8c5ab}"
BUILD_ROOT="${BUILD_ROOT:-/data/qwen38/build}"
COREX_BASE_IMAGE="${COREX_BASE_IMAGE:-}"
QWEN38_BASE_IMAGE="${QWEN38_BASE_IMAGE:-qwen38-bi100:corex3.2.3-text-0e899}"
IMAGE="${IMAGE:-qwen38-bi100:corex3.2.3-longctx-d972}"

if [[ -z "${COREX_BASE_IMAGE}" ]]; then
  echo "COREX_BASE_IMAGE must name the locally authorized CoreX 3.2.3 base image" >&2
  exit 2
fi
if [[ ! -d "${ADAPTER_REPO_DIR}/.git" ]]; then
  echo "community adapter repository not found: ${ADAPTER_REPO_DIR}" >&2
  exit 2
fi
if [[ ! -d "${BUILD_ROOT}" ]]; then
  echo "build root does not exist: ${BUILD_ROOT}" >&2
  exit 2
fi
if ! docker image inspect "${COREX_BASE_IMAGE}" >/dev/null 2>&1; then
  echo "CoreX base image is not available locally: ${COREX_BASE_IMAGE}" >&2
  exit 2
fi
if ! docker image inspect "${QWEN38_BASE_IMAGE}" >/dev/null 2>&1; then
  echo "verified Qwen3.8 base image is not available locally: ${QWEN38_BASE_IMAGE}" >&2
  exit 2
fi

actual_commit="$(cd "${ADAPTER_REPO_DIR}" && git rev-parse "${ADAPTER_COMMIT}^{commit}")"
if [[ "${actual_commit}" != "${ADAPTER_COMMIT}" ]]; then
  echo "adapter commit mismatch: ${actual_commit}" >&2
  exit 1
fi

work_dir="$(mktemp -d "${BUILD_ROOT%/}/qwen38-longctx.XXXXXX")"
cleanup() {
  rm -rf -- "${work_dir}"
}
trap cleanup EXIT

(cd "${ADAPTER_REPO_DIR}" && git archive "${ADAPTER_COMMIT}" qwen3_6_scripts) \
  | tar -x -C "${work_dir}"
(
  cd "${work_dir}"
  git init --quiet
  git apply --check "${PROJECT_ROOT}/patches/d972854-long-context.patch"
  git apply "${PROJECT_ROOT}/patches/d972854-long-context.patch"
)

if LC_ALL=C grep -R -n --include='*.py' '/tmp/vllm_decode_debug\.log' \
  "${work_dir}/qwen3_6_scripts"; then
  echo "debug log write remains in build context" >&2
  exit 1
fi

docker build \
  --build-arg "COREX_BASE_IMAGE=${COREX_BASE_IMAGE}" \
  --build-arg "QWEN38_BASE_IMAGE=${QWEN38_BASE_IMAGE}" \
  -t "${IMAGE}" \
  -f "${PROJECT_ROOT}/docker/Dockerfile.long-context" \
  "${work_dir}"

echo "built ${IMAGE} from dense adapter ${ADAPTER_COMMIT}"
