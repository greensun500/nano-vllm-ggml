cmake_minimum_required(VERSION 3.19)

project(nanovllm_native_runtime LANGUAGES C CXX)

include(CTest)

option(NANOVLLM_NATIVE_VULKAN "Build the native runtime with the GGML Vulkan backend" OFF)
option(NANOVLLM_NATIVE_CUDA "Build the native runtime with the GGML CUDA backend" OFF)
set(
    NANOVLLM_LLAMA_CPP_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
    CACHE PATH
    "Pinned llama.cpp source tree; only its ggml subdirectory is compiled"
)
# The embedded GGML tree is the official b9986 revision plus nano-vLLM's
# reviewed Mali shape-routing patch.  Pin the actual gitlink so a fresh build
# cannot silently compile a different tree, while reporting the official base
# separately for upstream traceability.
set(NANOVLLM_GGML_BASE_COMMIT "91c631b21d6e5d09e9c6659efdf6baeef5a44ddb")
set(NANOVLLM_GGML_COMMIT "5ed33380b4679533243ca45e172804d5ddfe59ec")

if(NOT EXISTS "${NANOVLLM_LLAMA_CPP_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "GGML sources are missing at ${NANOVLLM_LLAMA_CPP_DIR}/ggml. "
        "Run: git submodule update --init --recursive")
endif()

find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${NANOVLLM_LLAMA_CPP_DIR}" rev-parse HEAD
    RESULT_VARIABLE NANOVLLM_GGML_GIT_RESULT
    OUTPUT_VARIABLE NANOVLLM_GGML_ACTUAL_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT NANOVLLM_GGML_GIT_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Cannot determine the llama.cpp revision at ${NANOVLLM_LLAMA_CPP_DIR}; "
        "the native runtime requires the pinned official source checkout")
endif()
if(NOT "${NANOVLLM_GGML_ACTUAL_COMMIT}" STREQUAL "${NANOVLLM_GGML_COMMIT}")
    message(FATAL_ERROR
        "llama.cpp revision mismatch: expected ${NANOVLLM_GGML_COMMIT}, "
        "found ${NANOVLLM_GGML_ACTUAL_COMMIT}")
endif()

# GGML is statically embedded into the nano-vLLM Python extension.  It is not
# loaded as a separate llama.cpp runtime library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${NANOVLLM_NATIVE_CUDA} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${NANOVLLM_NATIVE_VULKAN} CACHE BOOL "" FORCE)

add_subdirectory(
    "${NANOVLLM_LLAMA_CPP_DIR}/ggml"
    "${CMAKE_CURRENT_BINARY_DIR}/ggml"
    EXCLUDE_FROM_ALL
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

Python3_add_library(_C MODULE WITH_SOABI
    csrc/native_module.cpp
    csrc/models/qwen35/graph.cpp
    csrc/models/qwen35/ops.cpp
    csrc/models/qwen35/weights.cpp
    csrc/python/qwen35_runtime_binding.cpp
    csrc/runtime/backend.cpp
    csrc/runtime/graph_executor.cpp
    csrc/runtime/gguf_loader.cpp
    csrc/runtime/paged_kv.cpp
    csrc/runtime/qwen35_model.cpp
    csrc/runtime/qwen35_runtime.cpp
    csrc/runtime/recurrent_state.cpp
)
target_compile_features(_C PRIVATE cxx_std_17)
target_compile_options(_C PRIVATE -Wall -Wextra)
target_include_directories(_C PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
target_link_libraries(_C PRIVATE ggml::ggml)
target_compile_definitions(_C PRIVATE
    NANOVLLM_GGML_COMMIT="${NANOVLLM_GGML_COMMIT}"
    NANOVLLM_GGML_BASE_COMMIT="${NANOVLLM_GGML_BASE_COMMIT}"
    NANOVLLM_NATIVE_HAS_VULKAN=$<BOOL:${NANOVLLM_NATIVE_VULKAN}>
    NANOVLLM_NATIVE_HAS_CUDA=$<BOOL:${NANOVLLM_NATIVE_CUDA}>
)
set_target_properties(_C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/nanovllm"
)

if(BUILD_TESTING)
    add_executable(nanovllm_graph_executor_test
        tests/native/test_graph_executor.cpp
        csrc/runtime/backend.cpp
        csrc/runtime/graph_executor.cpp
    )
    target_compile_features(nanovllm_graph_executor_test PRIVATE cxx_std_17)
    target_compile_options(nanovllm_graph_executor_test PRIVATE -Wall -Wextra)
    target_include_directories(
        nanovllm_graph_executor_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    )
    target_link_libraries(nanovllm_graph_executor_test PRIVATE ggml::ggml)
    target_compile_definitions(nanovllm_graph_executor_test PRIVATE
        NANOVLLM_NATIVE_HAS_VULKAN=$<BOOL:${NANOVLLM_NATIVE_VULKAN}>
        NANOVLLM_NATIVE_HAS_CUDA=$<BOOL:${NANOVLLM_NATIVE_CUDA}>
    )
    add_test(NAME graph_executor_placement COMMAND nanovllm_graph_executor_test)
endif()
