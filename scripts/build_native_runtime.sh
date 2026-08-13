#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENABLE_VULKAN="${NANOVLLM_NATIVE_VULKAN:-OFF}"
ENABLE_CUDA="${NANOVLLM_NATIVE_CUDA:-OFF}"
ENABLE_CUDA_GRAPHS="${NANOVLLM_NATIVE_CUDA_GRAPHS:-OFF}"
BUILD_KIND="cpu"
if [[ "${ENABLE_VULKAN}" == "ON" || "${ENABLE_VULKAN}" == "1" ]]; then
    ENABLE_VULKAN="ON"
    BUILD_KIND="vulkan"
else
    ENABLE_VULKAN="OFF"
fi
if [[ "${ENABLE_CUDA}" == "ON" || "${ENABLE_CUDA}" == "1" ]]; then
    ENABLE_CUDA="ON"
    if [[ "${BUILD_KIND}" != "cpu" ]]; then
        echo "nano-vLLM: build either Vulkan or CUDA at a time" >&2
        exit 2
    fi
    BUILD_KIND="cuda"
else
    ENABLE_CUDA="OFF"
fi
if [[ "${ENABLE_CUDA_GRAPHS}" == "ON" || "${ENABLE_CUDA_GRAPHS}" == "1" ]]; then
    ENABLE_CUDA_GRAPHS="ON"
else
    ENABLE_CUDA_GRAPHS="OFF"
fi
if [[ "${ENABLE_CUDA_GRAPHS}" == "ON" && "${ENABLE_CUDA}" != "ON" ]]; then
    echo "nano-vLLM: NANOVLLM_NATIVE_CUDA_GRAPHS requires NANOVLLM_NATIVE_CUDA=ON" >&2
    exit 2
fi
BUILD_DIR="${NANOVLLM_NATIVE_BUILD_DIR:-${ROOT_DIR}/build/native-${BUILD_KIND}}"
RUN_TESTS="${NANOVLLM_NATIVE_RUN_TESTS:-ON}"
if [[ "${RUN_TESTS}" == "1" ]]; then
    RUN_TESTS="ON"
elif [[ "${RUN_TESTS}" == "0" ]]; then
    RUN_TESTS="OFF"
fi

# GCC 12 on recent heterogeneous Arm systems can accept -mcpu=native while
# silently targeting only the baseline ISA.  GGML's follow-up probes then try
# invalid spellings such as -mcpu=native+dotprod, leaving the Q4 kernels without
# dot-product or i8mm support.  Select a conservative architecture explicitly
# when every prerequisite is visible.  Explicit CMake arguments always win.
AUTO_CMAKE_ARGS=()
ARM_ARCH_MODE="${NANOVLLM_NATIVE_ARM_ARCH:-auto}"
HAS_EXPLICIT_ARM_CONFIG=0
for arg in "$@"; do
    case "${arg}" in
        -DGGML_NATIVE=*|-DGGML_CPU_ARM_ARCH=*)
            HAS_EXPLICIT_ARM_CONFIG=1
            ;;
    esac
done

HOST_ARCH="$(uname -m)"
arm_feature_on_all_cpus() {
    local feature=$1
    awk -v feature="${feature}" '
        $1 == "Features" {
            seen = 1
            found = 0
            for (field = 3; field <= NF; ++field) {
                if ($field == feature) {
                    found = 1
                }
            }
            if (!found) {
                missing = 1
            }
        }
        END { exit !(seen && !missing) }
    ' /proc/cpuinfo
}

if [[ "${HAS_EXPLICIT_ARM_CONFIG}" == "0" && \
      ( "${HOST_ARCH}" == "aarch64" || "${HOST_ARCH}" == "arm64" ) ]]; then
    case "${ARM_ARCH_MODE}" in
        off|OFF|0|none|NONE)
            ;;
        auto|AUTO)
            AUTO_ARM_ARCH="armv8.6-a+dotprod+i8mm"
            ARM_CC="${CC:-cc}"
            if [[ -r /proc/cpuinfo ]] && \
               arm_feature_on_all_cpus asimddp && \
               arm_feature_on_all_cpus i8mm && \
               printf 'int main(void) { return 0; }\n' | \
                   "${ARM_CC}" -x c -c -o /dev/null \
                       "-march=${AUTO_ARM_ARCH}" - >/dev/null 2>&1; then
                AUTO_CMAKE_ARGS+=(
                    -DGGML_NATIVE=OFF
                    "-DGGML_CPU_ARM_ARCH=${AUTO_ARM_ARCH}"
                )
                echo "nano-vLLM: enabling Arm dotprod+i8mm kernels (${AUTO_ARM_ARCH})"
            else
                echo "nano-vLLM: Arm dotprod+i8mm auto-detection unavailable; using GGML defaults"
            fi
            ;;
        *)
            AUTO_CMAKE_ARGS+=(
                -DGGML_NATIVE=OFF
                "-DGGML_CPU_ARM_ARCH=${ARM_ARCH_MODE}"
            )
            echo "nano-vLLM: using requested Arm architecture ${ARM_ARCH_MODE}"
            ;;
    esac
fi

if [[ "${ENABLE_VULKAN}" == "ON" && -n "${NANOVLLM_VULKAN_GLSLC:-}" ]]; then
    if [[ ! -x "${NANOVLLM_VULKAN_GLSLC}" ]]; then
        echo "nano-vLLM: NANOVLLM_VULKAN_GLSLC is not executable: ${NANOVLLM_VULKAN_GLSLC}" >&2
        exit 2
    fi
    AUTO_CMAKE_ARGS+=(
        "-DVulkan_GLSLC_EXECUTABLE=${NANOVLLM_VULKAN_GLSLC}"
    )
    echo "nano-vLLM: using Vulkan shader compiler ${NANOVLLM_VULKAN_GLSLC}"
fi

cmake \
    -S "${ROOT_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING="${RUN_TESTS}" \
    -DNANOVLLM_NATIVE_VULKAN="${ENABLE_VULKAN}" \
    -DNANOVLLM_NATIVE_CUDA="${ENABLE_CUDA}" \
    -DNANOVLLM_NATIVE_CUDA_GRAPHS="${ENABLE_CUDA_GRAPHS}" \
    "${AUTO_CMAKE_ARGS[@]}" \
    "$@"
cmake --build "${BUILD_DIR}" --target _C --parallel "${NANOVLLM_BUILD_JOBS:-$(nproc)}"

if [[ "${RUN_TESTS}" == "ON" ]]; then
    cmake --build "${BUILD_DIR}" --target nanovllm_graph_executor_test \
        --parallel "${NANOVLLM_BUILD_JOBS:-$(nproc)}"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
fi

PYTHONPATH="${ROOT_DIR}" python3 -c \
    'from nanovllm.backends.native import build_info, available_backends; print(build_info()); print(available_backends())'
