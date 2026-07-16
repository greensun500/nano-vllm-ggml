#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENABLE_VULKAN="${NANOVLLM_NATIVE_VULKAN:-OFF}"
BUILD_KIND="cpu"
if [[ "${ENABLE_VULKAN}" == "ON" || "${ENABLE_VULKAN}" == "1" ]]; then
    ENABLE_VULKAN="ON"
    BUILD_KIND="vulkan"
else
    ENABLE_VULKAN="OFF"
fi
BUILD_DIR="${NANOVLLM_NATIVE_BUILD_DIR:-${ROOT_DIR}/build/native-${BUILD_KIND}}"
RUN_TESTS="${NANOVLLM_NATIVE_RUN_TESTS:-ON}"
if [[ "${RUN_TESTS}" == "1" ]]; then
    RUN_TESTS="ON"
elif [[ "${RUN_TESTS}" == "0" ]]; then
    RUN_TESTS="OFF"
fi

cmake \
    -S "${ROOT_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING="${RUN_TESTS}" \
    -DNANOVLLM_NATIVE_VULKAN="${ENABLE_VULKAN}" \
    "$@"
cmake --build "${BUILD_DIR}" --target _C --parallel "${NANOVLLM_BUILD_JOBS:-$(nproc)}"

if [[ "${RUN_TESTS}" == "ON" ]]; then
    cmake --build "${BUILD_DIR}" --target nanovllm_graph_executor_test \
        --parallel "${NANOVLLM_BUILD_JOBS:-$(nproc)}"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

PYTHONPATH="${ROOT_DIR}" python3 -c \
    'from nanovllm.backends.native import build_info, available_backends; print(build_info()); print(available_backends())'
