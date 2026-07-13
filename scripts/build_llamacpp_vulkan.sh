#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/cix/nano-vllm/llama.cpp-qwen35}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_CPP_DIR}/build_nanovllm_vulkan}"
VULKAN_INCLUDE_DIR="${VULKAN_INCLUDE_DIR:-/usr/include}"
VULKAN_LIBRARY="${VULKAN_LIBRARY:-/usr/lib/aarch64-linux-gnu/libvulkan.so}"
GLSLC="${GLSLC:-/usr/bin/glslc}"

cmake -S "${LLAMA_CPP_DIR}" -B "${BUILD_DIR}" \
  -DLLAMA_CURL=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_VULKAN=ON \
  -DVulkan_LIBRARY="${VULKAN_LIBRARY}" \
  -DVulkan_INCLUDE_DIR="${VULKAN_INCLUDE_DIR}" \
  -DVulkan_GLSLC_EXECUTABLE="${GLSLC}" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=OFF \
  -DGGML_AVX2=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_FMA=OFF \
  -DGGML_F16C=OFF \
  -DGGML_CPU_ARM_ARCH=armv9-a+sve \
  -DCMAKE_SYSTEM_PROCESSOR=armv9-a \
  -DCMAKE_SYSTEM_NAME=Linux

TARGETS="${TARGETS:-nanollama_backend}"
cmake --build "${BUILD_DIR}" --target ${TARGETS} -j"$(nproc)"

echo "Built: ${BUILD_DIR}/bin/libnanollama_backend.so"
