#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ADAPTER_REPO="${ADAPTER_REPO:-https://dev.modelhub.org.cn/icer/qwen36_01.git}"
ADAPTER_COMMIT="${ADAPTER_COMMIT:-0e89906481e9a6cb2475925adb41517676e12903}"
IMAGE="${IMAGE:-qwen38-bi100:corex3.2.3-text-0e899}"
BUILD_ROOT="${BUILD_ROOT:-/data/qwen38/build}"
BASE_IMAGE="${BASE_IMAGE:-}"

if [[ -z "${BASE_IMAGE}" ]]; then
  echo "BASE_IMAGE must name an already-authorized, locally available CoreX 3.2.3 LLM image" >&2
  exit 2
fi
if [[ ! -d "${BUILD_ROOT}" ]]; then
  echo "build root does not exist: ${BUILD_ROOT}" >&2
  exit 2
fi

work_dir="$(mktemp -d "${BUILD_ROOT%/}/qwen38-adapter.XXXXXX")"
cleanup() {
  rm -rf -- "${work_dir}"
}
trap cleanup EXIT

git -C "${work_dir}" init
git -C "${work_dir}" remote add origin "${ADAPTER_REPO}"
git -C "${work_dir}" fetch --depth=1 origin "${ADAPTER_COMMIT}"
git -C "${work_dir}" checkout --detach FETCH_HEAD

actual_commit="$(git -C "${work_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${ADAPTER_COMMIT}" ]]; then
  echo "adapter commit mismatch: ${actual_commit}" >&2
  exit 1
fi
test -x "${work_dir}/qwen3_6_scripts/patch_ops.sh"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE}" \
  -f "${PROJECT_ROOT}/docker/Dockerfile.qwen38" \
  "${work_dir}"

echo "built ${IMAGE} from adapter ${ADAPTER_COMMIT}"
