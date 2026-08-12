#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "runtime/backend.h"
#include "runtime/gguf_loader.h"
#include "runtime/qwen35_model.h"
#include "python/qwen35_runtime_binding.h"

#ifndef NANOVLLM_NATIVE_HAS_VULKAN
#define NANOVLLM_NATIVE_HAS_VULKAN 0
#endif
#ifndef NANOVLLM_NATIVE_HAS_CUDA
#define NANOVLLM_NATIVE_HAS_CUDA 0
#endif

#if NANOVLLM_NATIVE_HAS_VULKAN
#include "ggml-vulkan.h"
#endif
#if NANOVLLM_NATIVE_HAS_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

class BackendHandle {
public:
    explicit BackendHandle(ggml_backend_t value = nullptr) : value_(value) {}
    ~BackendHandle() {
        if (value_ != nullptr) {
            ggml_backend_free(value_);
        }
    }
    BackendHandle(const BackendHandle &) = delete;
    BackendHandle & operator=(const BackendHandle &) = delete;
    ggml_backend_t get() const { return value_; }

private:
    ggml_backend_t value_;
};

class ContextHandle {
public:
    explicit ContextHandle(ggml_context * value = nullptr) : value_(value) {}
    ~ContextHandle() {
        if (value_ != nullptr) {
            ggml_free(value_);
        }
    }
    ContextHandle(const ContextHandle &) = delete;
    ContextHandle & operator=(const ContextHandle &) = delete;
    ggml_context * get() const { return value_; }

private:
    ggml_context * value_;
};

class BufferHandle {
public:
    explicit BufferHandle(ggml_backend_buffer_t value = nullptr) : value_(value) {}
    ~BufferHandle() {
        if (value_ != nullptr) {
            ggml_backend_buffer_free(value_);
        }
    }
    BufferHandle(const BufferHandle &) = delete;
    BufferHandle & operator=(const BufferHandle &) = delete;
    ggml_backend_buffer_t get() const { return value_; }

private:
    ggml_backend_buffer_t value_;
};

class GgufHandle {
public:
    explicit GgufHandle(gguf_context * value = nullptr) : value_(value) {}
    ~GgufHandle() {
        if (value_ != nullptr) {
            gguf_free(value_);
        }
    }
    GgufHandle(const GgufHandle &) = delete;
    GgufHandle & operator=(const GgufHandle &) = delete;
    gguf_context * get() const { return value_; }

private:
    gguf_context * value_;
};

bool dict_set_owned(PyObject * dict, const char * key, PyObject * value) {
    if (value == nullptr) {
        return false;
    }
    const int result = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return result == 0;
}

PyObject * backend_record(const char * kind, const char * name, int index) {
    return Py_BuildValue("{s:s,s:s,s:i}", "kind", kind, "name", name, "index", index);
}

PyObject * py_build_info(PyObject *, PyObject *) {
    PyObject * info = PyDict_New();
    if (info == nullptr) {
        return nullptr;
    }
    const bool ok =
        dict_set_owned(info, "runtime", PyUnicode_FromString("nanovllm_native")) &&
        dict_set_owned(info, "abi_version", PyLong_FromLong(1)) &&
        dict_set_owned(info, "ggml_commit", PyUnicode_FromString(NANOVLLM_GGML_COMMIT)) &&
        dict_set_owned(
            info, "ggml_base_commit", PyUnicode_FromString(NANOVLLM_GGML_BASE_COMMIT)) &&
        dict_set_owned(info, "cpu", PyBool_FromLong(1)) &&
        dict_set_owned(info, "vulkan", PyBool_FromLong(NANOVLLM_NATIVE_HAS_VULKAN)) &&
        dict_set_owned(info, "cuda", PyBool_FromLong(NANOVLLM_NATIVE_HAS_CUDA)) &&
        dict_set_owned(info, "persistent_cpu_threadpool", PyBool_FromLong(1)) &&
        dict_set_owned(info, "uses_llama_context", PyBool_FromLong(0));
    if (!ok) {
        Py_DECREF(info);
        return nullptr;
    }
    return info;
}

PyObject * py_available_backends(PyObject *, PyObject *) {
    PyObject * result = PyList_New(0);
    if (result == nullptr) {
        return nullptr;
    }

    {
        BackendHandle cpu(ggml_backend_cpu_init());
        if (cpu.get() == nullptr) {
            Py_DECREF(result);
            PyErr_SetString(PyExc_RuntimeError, "GGML CPU backend initialization failed");
            return nullptr;
        }
        PyObject * record = backend_record("cpu", ggml_backend_name(cpu.get()), 0);
        if (record == nullptr || PyList_Append(result, record) < 0) {
            Py_XDECREF(record);
            Py_DECREF(result);
            return nullptr;
        }
        Py_DECREF(record);
    }

#if NANOVLLM_NATIVE_HAS_VULKAN
    const int count = ggml_backend_vk_get_device_count();
    for (int index = 0; index < count; ++index) {
        std::array<char, 256> description{};
        ggml_backend_vk_get_device_description(index, description.data(), description.size());
        PyObject * record = backend_record("vulkan", description.data(), index);
        if (record == nullptr || PyList_Append(result, record) < 0) {
            Py_XDECREF(record);
            Py_DECREF(result);
            return nullptr;
        }
        Py_DECREF(record);
    }
#endif
#if NANOVLLM_NATIVE_HAS_CUDA
    const int cuda_count = ggml_backend_cuda_get_device_count();
    for (int index = 0; index < cuda_count; ++index) {
        std::array<char, 256> description{};
        ggml_backend_cuda_get_device_description(index, description.data(), description.size());
        PyObject * record = backend_record("cuda", description.data(), index);
        if (record == nullptr || PyList_Append(result, record) < 0) {
            Py_XDECREF(record);
            Py_DECREF(result);
            return nullptr;
        }
        Py_DECREF(record);
    }
#endif
    return result;
}

const char * gguf_string(const gguf_context * ctx, const char * key) {
    const int64_t index = gguf_find_key(ctx, key);
    if (index < 0 || gguf_get_kv_type(ctx, index) != GGUF_TYPE_STRING) {
        return nullptr;
    }
    return gguf_get_val_str(ctx, index);
}

PyObject * py_gguf_info(PyObject *, PyObject * args, PyObject * kwargs) {
    const char * path = nullptr;
    static const char * keywords[] = {"path", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:gguf_info", const_cast<char **>(keywords), &path)) {
        return nullptr;
    }

    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    GgufHandle gguf(gguf_init_from_file(path, params));
    if (gguf.get() == nullptr) {
        PyErr_Format(PyExc_ValueError, "failed to read GGUF metadata from %s", path);
        return nullptr;
    }

    PyObject * info = PyDict_New();
    if (info == nullptr) {
        return nullptr;
    }
    const char * architecture = gguf_string(gguf.get(), "general.architecture");
    const char * name = gguf_string(gguf.get(), "general.name");
    const bool ok =
        dict_set_owned(info, "path", PyUnicode_FromString(path)) &&
        dict_set_owned(info, "version", PyLong_FromUnsignedLong(gguf_get_version(gguf.get()))) &&
        dict_set_owned(info, "alignment", PyLong_FromSize_t(gguf_get_alignment(gguf.get()))) &&
        dict_set_owned(info, "metadata_count", PyLong_FromLongLong(gguf_get_n_kv(gguf.get()))) &&
        dict_set_owned(info, "tensor_count", PyLong_FromLongLong(gguf_get_n_tensors(gguf.get()))) &&
        dict_set_owned(info, "architecture", architecture != nullptr ? PyUnicode_FromString(architecture) : Py_NewRef(Py_None)) &&
        dict_set_owned(info, "name", name != nullptr ? PyUnicode_FromString(name) : Py_NewRef(Py_None));
    if (!ok) {
        Py_DECREF(info);
        return nullptr;
    }
    return info;
}

ggml_backend_t create_backend(const char * kind, int threads) {
    if (std::strcmp(kind, "cpu") == 0) {
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend != nullptr) {
            ggml_backend_cpu_set_n_threads(backend, threads);
        }
        return backend;
    }
#if NANOVLLM_NATIVE_HAS_VULKAN
    if (std::strcmp(kind, "vulkan") == 0) {
        if (ggml_backend_vk_get_device_count() <= 0) {
            PyErr_SetString(PyExc_RuntimeError, "native runtime was built with Vulkan but no Vulkan device is available");
            return nullptr;
        }
        return ggml_backend_vk_init(0);
    }
#endif
#if NANOVLLM_NATIVE_HAS_CUDA
    if (std::strcmp(kind, "cuda") == 0) {
        if (ggml_backend_cuda_get_device_count() <= 0) {
            PyErr_SetString(PyExc_RuntimeError, "native runtime was built with CUDA but no CUDA device is available");
            return nullptr;
        }
        return ggml_backend_cuda_init(0);
    }
#endif
    PyErr_Format(
        PyExc_ValueError,
        "backend must be 'cpu', 'vulkan', or 'cuda'; got %s",
        kind);
    return nullptr;
}

PyObject * py_matmul_smoke(PyObject *, PyObject * args, PyObject * kwargs) {
    const char * backend_kind = "cpu";
    int threads = 1;
    static const char * keywords[] = {"backend", "threads", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|si:matmul_smoke", const_cast<char **>(keywords), &backend_kind, &threads)) {
        return nullptr;
    }
    if (threads <= 0) {
        PyErr_SetString(PyExc_ValueError, "threads must be positive");
        return nullptr;
    }

    BackendHandle backend(create_backend(backend_kind, threads));
    if (backend.get() == nullptr) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "failed to initialize the %s backend", backend_kind);
        }
        return nullptr;
    }

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ContextHandle ctx(ggml_init(params));
    if (ctx.get() == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "failed to create GGML metadata context");
        return nullptr;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 2);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_tensor * output = ggml_mul_mat(ctx.get(), weights, input);
    ggml_set_name(weights, "smoke.weights");
    ggml_set_name(input, "smoke.input");
    ggml_set_name(output, "smoke.output");
    ggml_set_input(weights);
    ggml_set_input(input);
    ggml_set_output(output);

    if (!ggml_backend_supports_op(backend.get(), output)) {
        PyErr_Format(PyExc_RuntimeError, "%s backend does not support the F32 mul_mat smoke graph", backend_kind);
        return nullptr;
    }

    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (buffer.get() == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate GGML backend tensors");
        return nullptr;
    }

    const std::array<float, 8> weights_data{1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<float, 12> input_data{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    const std::array<float, 6> expected{1, 5, 2, 6, 3, 7};
    ggml_backend_tensor_set(weights, weights_data.data(), 0, sizeof(weights_data));
    ggml_backend_tensor_set(input, input_data.data(), 0, sizeof(input_data));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "GGML graph failed: %s", ggml_status_to_string(status));
        return nullptr;
    }
    ggml_backend_synchronize(backend.get());

    std::array<float, 6> actual{};
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs_error = std::max(max_abs_error, std::abs(actual[i] - expected[i]));
    }

    PyObject * values = PyList_New(actual.size());
    if (values == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        PyList_SET_ITEM(values, i, PyFloat_FromDouble(actual[i]));
    }
    PyObject * result = PyDict_New();
    if (result == nullptr) {
        Py_DECREF(values);
        return nullptr;
    }
    const bool ok =
        dict_set_owned(result, "backend", PyUnicode_FromString(backend_kind)) &&
        dict_set_owned(result, "output", values) &&
        dict_set_owned(result, "max_abs_error", PyFloat_FromDouble(max_abs_error));
    if (!ok) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}

PyObject * py_q4_0_matmul_smoke(PyObject *, PyObject * args, PyObject * kwargs) {
    const char * backend_kind = "cpu";
    int threads = 1;
    static const char * keywords[] = {"backend", "threads", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|si:q4_0_matmul_smoke", const_cast<char **>(keywords), &backend_kind, &threads)) {
        return nullptr;
    }
    if (threads <= 0) {
        PyErr_SetString(PyExc_ValueError, "threads must be positive");
        return nullptr;
    }

    BackendHandle backend(create_backend(backend_kind, threads));
    if (backend.get() == nullptr) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "failed to initialize the %s backend", backend_kind);
        }
        return nullptr;
    }

    constexpr int64_t k = 32;
    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ContextHandle ctx(ggml_init(params));
    if (ctx.get() == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "failed to create GGML metadata context");
        return nullptr;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_0, k, m);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, n);
    ggml_tensor * output = ggml_mul_mat(ctx.get(), weights, input);
    ggml_set_name(weights, "smoke.q4_0_weights");
    ggml_set_name(input, "smoke.input");
    ggml_set_name(output, "smoke.output");
    ggml_set_input(weights);
    ggml_set_input(input);
    ggml_set_output(output);

    if (!ggml_backend_supports_op(backend.get(), output)) {
        PyErr_Format(PyExc_RuntimeError, "%s backend does not support Q4_0 mul_mat", backend_kind);
        return nullptr;
    }
    BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (buffer.get() == nullptr) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate Q4_0 smoke tensors");
        return nullptr;
    }

    std::array<float, k * m> weights_f32{};
    for (int64_t i = 0; i < k; ++i) {
        weights_f32[i] = static_cast<float>(i) - 15.5f;
        weights_f32[k + i] = static_cast<float>(i % 8) - 3.5f;
    }
    const size_t quantized_size = ggml_row_size(GGML_TYPE_Q4_0, k) * m;
    std::vector<uint8_t> weights_q4(quantized_size);
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(), 0, m, k, nullptr);
    if (written != quantized_size) {
        PyErr_Format(
            PyExc_RuntimeError,
            "Q4_0 quantizer wrote %zu bytes, expected %zu",
            written,
            quantized_size);
        return nullptr;
    }

    std::array<float, k * n> input_data{};
    input_data[0] = 1.0f;
    input_data[k + 1] = 1.0f;
    input_data[2 * k + 2] = 1.0f;
    ggml_backend_tensor_set(weights, weights_q4.data(), 0, weights_q4.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, sizeof(input_data));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "Q4_0 GGML graph failed: %s", ggml_status_to_string(status));
        return nullptr;
    }
    ggml_backend_synchronize(backend.get());

    std::array<float, k * m> dequantized{};
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, k);
    for (int64_t row = 0; row < m; ++row) {
        traits->to_float(weights_q4.data() + row * row_size, dequantized.data() + row * k, k);
    }
    const std::array<float, m * n> expected{
        dequantized[0], dequantized[k],
        dequantized[1], dequantized[k + 1],
        dequantized[2], dequantized[k + 2],
    };

    std::array<float, m * n> actual{};
    ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs_error = std::max(max_abs_error, std::abs(actual[i] - expected[i]));
    }

    PyObject * values = PyList_New(actual.size());
    if (values == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        PyList_SET_ITEM(values, i, PyFloat_FromDouble(actual[i]));
    }
    PyObject * result = PyDict_New();
    if (result == nullptr) {
        Py_DECREF(values);
        return nullptr;
    }
    const bool ok =
        dict_set_owned(result, "backend", PyUnicode_FromString(backend_kind)) &&
        dict_set_owned(result, "weight_type", PyUnicode_FromString("Q4_0")) &&
        dict_set_owned(result, "output", values) &&
        dict_set_owned(result, "max_abs_error", PyFloat_FromDouble(max_abs_error));
    if (!ok) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}

PyObject * py_qwen35_weight_load_smoke(PyObject *, PyObject * args, PyObject * kwargs) {
    const char * path = nullptr;
    const char * backend_kind = "cpu";
    int threads = 1;
    static const char * keywords[] = {"path", "backend", "threads", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "s|si:qwen35_weight_load_smoke",
            const_cast<char **>(keywords),
            &path,
            &backend_kind,
            &threads)) {
        return nullptr;
    }
    if (threads <= 0) {
        PyErr_SetString(PyExc_ValueError, "threads must be positive");
        return nullptr;
    }

    try {
        using nanovllm::native::BackendKind;
        using nanovllm::native::BackendList;
        using nanovllm::native::GgufWeights;

        const BackendKind kind = nanovllm::native::parse_backend_kind(backend_kind);
        BackendList backends = kind == BackendKind::Cpu
            ? BackendList::cpu_only(threads)
            : kind == BackendKind::Vulkan
                ? BackendList::vulkan_with_cpu(threads)
                : BackendList::cuda_with_cpu(threads);
        auto weights = std::make_unique<GgufWeights>(path);
        const auto manifest = nanovllm::native::qwen35::validate(
            weights->metadata(), weights->tensor_context());
        weights->load(backends.at(0).get());
        nanovllm::native::assert_buffer_on_backend(
            weights->buffer(), backends.at(0).get(), "Qwen3.5 weights");
        nanovllm::native::assert_tensor_on_backend(
            weights->require_tensor("token_embd.weight"),
            backends.at(0).get(),
            "token_embd.weight");

        PyObject * result = PyDict_New();
        if (result == nullptr) {
            return nullptr;
        }
        const bool ok =
            dict_set_owned(result, "backend", PyUnicode_FromString(backend_kind)) &&
            dict_set_owned(
                result,
                "buffer_type",
                PyUnicode_FromString(nanovllm::native::buffer_type_name(weights->buffer_type()).c_str())) &&
            dict_set_owned(result, "tensor_count", PyLong_FromSize_t(weights->tensor_names().size())) &&
            dict_set_owned(result, "file_tensor_bytes", PyLong_FromSize_t(weights->file_tensor_bytes())) &&
            dict_set_owned(result, "resident_buffer_bytes", PyLong_FromSize_t(weights->resident_buffer_bytes())) &&
            dict_set_owned(
                result,
                "has_mtp",
                PyBool_FromLong(weights->contains("blk.24.nextn.eh_proj.weight"))) &&
            dict_set_owned(
                result,
                "strict_q4_0_profile",
                PyBool_FromLong(manifest.strict_current_profile()));
        if (!ok) {
            Py_DECREF(result);
            return nullptr;
        }
        return result;
    } catch (const std::exception & error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject * layer_indices(const nanovllm::native::qwen35::Config & config, bool recurrent) {
    PyObject * result = PyList_New(0);
    if (result == nullptr) {
        return nullptr;
    }
    for (uint32_t layer = 0; layer < config.main_layers; ++layer) {
        if (config.is_recurrent_layer(layer) != recurrent) {
            continue;
        }
        PyObject * value = PyLong_FromUnsignedLong(layer);
        if (value == nullptr || PyList_Append(result, value) < 0) {
            Py_XDECREF(value);
            Py_DECREF(result);
            return nullptr;
        }
        Py_DECREF(value);
    }
    return result;
}

PyObject * py_qwen35_model_info(PyObject *, PyObject * args, PyObject * kwargs) {
    const char * path = nullptr;
    static const char * keywords[] = {"path", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "s:qwen35_model_info", const_cast<char **>(keywords), &path)) {
        return nullptr;
    }
    try {
        const auto manifest = nanovllm::native::qwen35::inspect_file(path);
        const auto & config = manifest.config;
        PyObject * result = PyDict_New();
        if (result == nullptr) {
            return nullptr;
        }
        const bool ok =
            dict_set_owned(result, "architecture", PyUnicode_FromString(config.architecture.c_str())) &&
            dict_set_owned(result, "block_count", PyLong_FromUnsignedLong(config.block_count)) &&
            dict_set_owned(result, "main_layers", PyLong_FromUnsignedLong(config.main_layers)) &&
            dict_set_owned(
                result, "nextn_predict_layers", PyLong_FromUnsignedLong(config.nextn_predict_layers)) &&
            dict_set_owned(result, "context_length", PyLong_FromUnsignedLong(config.context_length)) &&
            dict_set_owned(result, "embedding_length", PyLong_FromUnsignedLong(config.embedding_length)) &&
            dict_set_owned(result, "feed_forward_length", PyLong_FromUnsignedLong(config.feed_forward_length)) &&
            dict_set_owned(result, "vocabulary_size", PyLong_FromUnsignedLongLong(config.vocabulary_size)) &&
            dict_set_owned(result, "tensor_count", PyLong_FromSize_t(manifest.tensors.size())) &&
            dict_set_owned(
                result, "strict_q4_0_profile", PyBool_FromLong(manifest.strict_current_profile())) &&
            dict_set_owned(result, "recurrent_layers", layer_indices(config, true)) &&
            dict_set_owned(result, "full_attention_layers", layer_indices(config, false));
        if (!ok) {
            Py_DECREF(result);
            return nullptr;
        }
        return result;
    } catch (const std::exception & error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"build_info", py_build_info, METH_NOARGS, "Return native-runtime build provenance."},
    {"available_backends", py_available_backends, METH_NOARGS, "Enumerate embedded GGML backends."},
    {"gguf_info", reinterpret_cast<PyCFunction>(py_gguf_info), METH_VARARGS | METH_KEYWORDS,
     "Read GGUF metadata without using llama_model."},
    {"matmul_smoke", reinterpret_cast<PyCFunction>(py_matmul_smoke), METH_VARARGS | METH_KEYWORDS,
     "Run a small GGML graph on an embedded CPU or Vulkan backend."},
    {"q4_0_matmul_smoke", reinterpret_cast<PyCFunction>(py_q4_0_matmul_smoke), METH_VARARGS | METH_KEYWORDS,
     "Run a quantized Q4_0 GGML graph on an embedded CPU or Vulkan backend."},
    {"qwen35_weight_load_smoke", reinterpret_cast<PyCFunction>(py_qwen35_weight_load_smoke),
     METH_VARARGS | METH_KEYWORDS,
     "Load all Qwen3.5 GGUF tensors into a persistent native backend buffer."},
    {"qwen35_model_info", reinterpret_cast<PyCFunction>(py_qwen35_model_info),
     METH_VARARGS | METH_KEYWORDS,
     "Validate the exact Qwen3.5-2B + bundled MTP GGUF contract."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_C",
    "nano-vLLM native GGML runtime (does not link the llama.cpp high-level runtime)",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__C() {
    PyObject * result = PyModule_Create(&module);
    if (result == nullptr) {
        return nullptr;
    }
    if (nanovllm::python::register_qwen35_runtime_type(result) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
