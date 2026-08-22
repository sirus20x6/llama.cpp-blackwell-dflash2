#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_JOBS="${BUILD_JOBS:-20}"

cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_NCCL=OFF \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_CUSTOM_RUNTIME_ABI=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${ROOT_DIR}/build" --target llama-server llama-custom-runtime-abi -j "${BUILD_JOBS}"

cmake -S "${ROOT_DIR}/runtime" -B "${ROOT_DIR}/runtime/build-blackwell" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed \
    -DGGML_CUDA_NCCL=OFF \
    -DGGML_NATIVE=ON \
    -DADVANCED_GGUF_QUANTIZER_BUILD_BENCH=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${ROOT_DIR}/runtime/build-blackwell" -j "${BUILD_JOBS}"
