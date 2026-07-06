#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/cix/nano-vlm/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_CPP_DIR}/build_nanovllm_cpu}"

cmake -S "${LLAMA_CPP_DIR}" -B "${BUILD_DIR}" \
  -DLLAMA_CURL=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=OFF \
  -DGGML_AVX2=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_FMA=OFF \
  -DGGML_F16C=OFF \
  -DGGML_CPU_ARM_ARCH=armv9-a+i8mm+dotprod+sve \
  -DGGML_CPU_KLEIDIAI=ON

TARGETS="${TARGETS:-nanollama_backend}"
cmake --build "${BUILD_DIR}" --target ${TARGETS} -j"$(nproc)"

echo "Built: ${BUILD_DIR}/bin/libnanollama_backend.so"
