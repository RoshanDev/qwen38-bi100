#!/usr/bin/env bash
set -uo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
MODEL_REVISION="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
MODEL_DIR="${MODEL_DIR:-/model}"
MAX_WORKERS="${MAX_WORKERS:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-10}"

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    echo "download attempt ${attempt}/${MAX_ATTEMPTS}"
    if hf download "${MODEL_ID}" \
        --revision "${MODEL_REVISION}" \
        --local-dir "${MODEL_DIR}" \
        --max-workers "${MAX_WORKERS}"; then
        echo "download complete"
        exit 0
    fi

    echo "download attempt ${attempt} failed; retrying in ${RETRY_DELAY_SECONDS}s" >&2
    sleep "${RETRY_DELAY_SECONDS}"
done

echo "download failed after ${MAX_ATTEMPTS} attempts" >&2
exit 1
