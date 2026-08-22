#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"

SERVER_BIN="${SERVER_BIN:-${ROOT_DIR}/build/bin/llama-server}"
CUSTOM_LIB_DIR="${CUSTOM_LIB_DIR:-${ROOT_DIR}/runtime/build-blackwell/bin}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/Qwen3.8-27B-heretic-ara-MXFP6-MXFP8.gguf}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-${MODEL_DIR}/Qwen3.8-27B-DFlash2-MXFP6_E2M3.gguf}"

LLAMA_THREADS="${LLAMA_THREADS:-24}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-262144}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-4096}"
LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-4096}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
CACHE_TYPE_K="${CACHE_TYPE_K:-mxfp8}"
CACHE_TYPE_V="${CACHE_TYPE_V:-mxfp8}"
DFLASH_DRAFT_MAX="${DFLASH_DRAFT_MAX:-3}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-heretic-ara-mxfp6-mxfp8-dflash2}"
LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLAMA_PORT="${LLAMA_PORT:-9080}"
LLAMA_API_KEY="${LLAMA_API_KEY:-local-vllm}"

export GGML_CUDA_MXFP6_FP8_ACTS="${GGML_CUDA_MXFP6_FP8_ACTS:-none}"
export GGML_CUDA_MXFP6_MMVQ="${GGML_CUDA_MXFP6_MMVQ:-auto}"
export GGML_CUDA_NVFP4_MMVQ="${GGML_CUDA_NVFP4_MMVQ:-auto}"
export GGML_CUDA_MXFP8_MMA="${GGML_CUDA_MXFP8_MMA:-1}"

required_files=(
    "${SERVER_BIN}"
    "${ROOT_DIR}/build/bin/libllama-custom-runtime-abi.so"
    "${CUSTOM_LIB_DIR}/libllama.so.0"
    "${CUSTOM_LIB_DIR}/libggml.so.0"
    "${CUSTOM_LIB_DIR}/libggml-cpu.so.0"
    "${CUSTOM_LIB_DIR}/libggml-cuda.so.0"
    "${CUSTOM_LIB_DIR}/libggml-base.so.0"
    "${MODEL_PATH}"
    "${DRAFT_MODEL_PATH}"
)

for required_file in "${required_files[@]}"; do
    if [[ ! -r "${required_file}" ]]; then
        printf 'Required file is not readable: %s\n' "${required_file}" >&2
        exit 1
    fi
done

custom_core_libs=(
    "${ROOT_DIR}/build/bin/libllama-custom-runtime-abi.so"
    "${CUSTOM_LIB_DIR}/libllama.so.0"
    "${CUSTOM_LIB_DIR}/libggml.so.0"
    "${CUSTOM_LIB_DIR}/libggml-cpu.so.0"
    "${CUSTOM_LIB_DIR}/libggml-cuda.so.0"
    "${CUSTOM_LIB_DIR}/libggml-base.so.0"
)
printf -v custom_core_preload ':%s' "${custom_core_libs[@]}"
custom_core_preload="${custom_core_preload:1}"
export LD_PRELOAD="${custom_core_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"

server_args=(
    --model "${MODEL_PATH}"
    --alias "${SERVED_MODEL_NAME}"
    --spec-draft-model "${DRAFT_MODEL_PATH}"
    --spec-type draft-dflash
    --spec-draft-n-max "${DFLASH_DRAFT_MAX}"
    --spec-draft-n-min 0
    --n-gpu-layers all
    --spec-draft-ngl all
    --spec-draft-device CUDA0
    --threads "${LLAMA_THREADS}"
    --parallel "${LLAMA_PARALLEL}"
    --ctx-size "${LLAMA_CTX_SIZE}"
    --batch-size "${LLAMA_BATCH_SIZE}"
    --ubatch-size "${LLAMA_UBATCH_SIZE}"
    --no-kv-unified
    --cache-type-k "${CACHE_TYPE_K}"
    --cache-type-v "${CACHE_TYPE_V}"
    --spec-draft-type-k f16
    --spec-draft-type-v f16
    --flash-attn on
    --fit off
    --load-mode mmap
    --jinja
    --host "${LLAMA_HOST}"
    --port "${LLAMA_PORT}"
    --api-key "${LLAMA_API_KEY}"
    --metrics
    --no-webui
)

printf 'Starting llama-server:'
printf ' %q' "${SERVER_BIN}" "${server_args[@]}" "$@"
printf '\n'

exec "${SERVER_BIN}" "${server_args[@]}" "$@"
