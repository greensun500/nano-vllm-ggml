#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "python/qwen35_runtime_binding.h"
#include "runtime/qwen35_runtime.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanovllm::python {
namespace {

using nanovllm::native::Qwen35ExecutionPlan;
using nanovllm::native::Qwen35AttentionImplementation;
using nanovllm::native::Qwen35ExecutionProfileStats;
using nanovllm::native::Qwen35GraphReuseStats;
using nanovllm::native::Qwen35MemoryStats;
using nanovllm::native::Qwen35MtpProfileStats;
using nanovllm::native::Qwen35MtpResult;
using nanovllm::native::Qwen35MaliMtpPrefillStrategy;
using nanovllm::native::Qwen35Runtime;
using nanovllm::native::Qwen35RuntimeOptions;

static_assert(sizeof(std::int32_t) == 4, "Qwen35Runtime requires 32-bit int32_t");
static_assert(sizeof(int) == 4, "the PEP 3118 'i' format must be 32 bits");

struct PyQwen35Runtime {
    PyObject_HEAD
    Qwen35Runtime * runtime;
};

// Py_BEGIN_ALLOW_THREADS cannot safely surround a throwing C++ call because
// its restore statement is skipped during stack unwinding.  This guard always
// reacquires the GIL before an exception is translated to a Python exception.
class AllowThreads {
public:
    AllowThreads() : thread_state_(PyEval_SaveThread()) {}
    ~AllowThreads() { PyEval_RestoreThread(thread_state_); }

    AllowThreads(const AllowThreads &) = delete;
    AllowThreads & operator=(const AllowThreads &) = delete;

private:
    PyThreadState * thread_state_;
};

class BufferView {
public:
    BufferView() = default;
    ~BufferView() {
        if (acquired_) {
            PyBuffer_Release(&view_);
        }
    }

    BufferView(const BufferView &) = delete;
    BufferView & operator=(const BufferView &) = delete;

    bool acquire_int32_vector(PyObject * object, const char * label) {
        constexpr int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS;
        if (PyObject_GetBuffer(object, &view_, flags) < 0) {
            PyErr_Clear();
            PyErr_Format(
                PyExc_TypeError,
                "Qwen35Runtime execution-plan field '%s' must expose a "
                "C-contiguous one-dimensional int32 buffer",
                label);
            return false;
        }
        acquired_ = true;

        if (view_.ndim != 1 || view_.shape == nullptr ||
            view_.itemsize != static_cast<Py_ssize_t>(sizeof(std::int32_t)) ||
            view_.format == nullptr || std::strcmp(view_.format, "i") != 0 ||
            !PyBuffer_IsContiguous(&view_, 'C')) {
            PyErr_Format(
                PyExc_TypeError,
                "Qwen35Runtime execution-plan field '%s' must be a "
                "C-contiguous one-dimensional signed int32 buffer",
                label);
            return false;
        }
        if (view_.shape[0] < 0 || view_.len < 0) {
            PyErr_Format(PyExc_ValueError, "Qwen35Runtime field '%s' has a negative buffer size", label);
            return false;
        }
        const auto elements = static_cast<std::size_t>(view_.shape[0]);
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) ||
            static_cast<std::size_t>(view_.len) != elements * sizeof(std::int32_t)) {
            PyErr_Format(PyExc_ValueError, "Qwen35Runtime field '%s' has an inconsistent buffer size", label);
            return false;
        }
        if (elements != 0 && view_.buf == nullptr) {
            PyErr_Format(PyExc_ValueError, "Qwen35Runtime field '%s' exposes a null data pointer", label);
            return false;
        }
        return true;
    }

    std::size_t size() const noexcept {
        return static_cast<std::size_t>(view_.shape[0]);
    }

    const std::int32_t * data() const noexcept {
        return static_cast<const std::int32_t *>(view_.buf);
    }

private:
    Py_buffer view_{};
    bool acquired_ = false;
};

void translate_cpp_exception() noexcept {
    try {
        throw;
    } catch (const std::bad_alloc &) {
        PyErr_NoMemory();
    } catch (const std::invalid_argument & error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const std::out_of_range & error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const std::exception & error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Qwen35Runtime failed with an unknown C++ exception");
    }
}

PyObject * required_dict_item(PyObject * dictionary, const char * key) {
    PyObject * value = PyDict_GetItemString(dictionary, key);  // borrowed
    if (value == nullptr) {
        PyErr_Format(PyExc_KeyError, "Qwen35Runtime execution plan is missing field '%s'", key);
    }
    return value;
}

bool parse_bool(PyObject * value, const char * label, bool & result) {
    if (!PyBool_Check(value)) {
        PyErr_Format(PyExc_TypeError, "Qwen35Runtime argument '%s' must be bool", label);
        return false;
    }
    result = value == Py_True;
    return true;
}

bool parse_attention_implementation(
    PyObject * value,
    Qwen35AttentionImplementation & result) {
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Qwen35Runtime argument 'attention_impl' must be str");
        return false;
    }
    const char * implementation = PyUnicode_AsUTF8(value);
    if (implementation == nullptr) {
        return false;
    }
    if (std::strcmp(implementation, "math") == 0) {
        result = Qwen35AttentionImplementation::Math;
        return true;
    }
    if (std::strcmp(implementation, "auto") == 0) {
        result = Qwen35AttentionImplementation::Auto;
        return true;
    }
    if (std::strcmp(implementation, "flash") == 0) {
        result = Qwen35AttentionImplementation::Flash;
        return true;
    }
    if (std::strcmp(implementation, "paged") == 0) {
        result = Qwen35AttentionImplementation::Paged;
        return true;
    }
    PyErr_SetString(
        PyExc_ValueError,
        "Qwen35Runtime argument 'attention_impl' must be 'math', 'auto', 'flash', or 'paged'");
    return false;
}

bool parse_mali_mtp_prefill_strategy(
    PyObject * value,
    Qwen35MaliMtpPrefillStrategy & result) {
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Qwen35Runtime argument 'mali_mtp_prefill_strategy' must be str");
        return false;
    }
    const char * strategy = PyUnicode_AsUTF8(value);
    if (strategy == nullptr) {
        return false;
    }
    if (std::strcmp(strategy, "whole") == 0) {
        result = Qwen35MaliMtpPrefillStrategy::WholePrompt;
        return true;
    }
    if (std::strcmp(strategy, "chunked_legacy") == 0) {
        result = Qwen35MaliMtpPrefillStrategy::ChunkedLegacy;
        return true;
    }
    if (std::strcmp(strategy, "chunked_staged") == 0) {
        result = Qwen35MaliMtpPrefillStrategy::ChunkedStaged;
        return true;
    }
    PyErr_SetString(
        PyExc_ValueError,
        "Qwen35Runtime argument 'mali_mtp_prefill_strategy' must be "
        "'whole', 'chunked_legacy', or 'chunked_staged'");
    return false;
}

bool parse_size(PyObject * value, const char * label, std::size_t & result) {
    if (!PyLong_Check(value) || PyBool_Check(value)) {
        PyErr_Format(PyExc_TypeError, "Qwen35Runtime argument '%s' must be a non-negative integer", label);
        return false;
    }
    const unsigned long long parsed = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        PyErr_Format(PyExc_ValueError, "Qwen35Runtime argument '%s' is outside the size_t range", label);
        return false;
    }
    if (parsed > std::numeric_limits<std::size_t>::max()) {
        PyErr_Format(PyExc_ValueError, "Qwen35Runtime argument '%s' is outside the size_t range", label);
        return false;
    }
    result = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_positive_size(PyObject * value, const char * label, std::size_t & result) {
    if (!parse_size(value, label, result)) {
        return false;
    }
    if (result == 0) {
        PyErr_Format(PyExc_ValueError, "Qwen35Runtime argument '%s' must be positive", label);
        return false;
    }
    return true;
}

bool parse_positive_int(PyObject * value, const char * label, int & result) {
    std::size_t parsed = 0;
    if (!parse_positive_size(value, label, parsed)) {
        return false;
    }
    if (parsed > static_cast<std::size_t>(INT_MAX)) {
        PyErr_Format(PyExc_ValueError, "Qwen35Runtime argument '%s' exceeds INT_MAX", label);
        return false;
    }
    result = static_cast<int>(parsed);
    return true;
}

bool parse_plan_int32(PyObject * dictionary, const char * key, std::int32_t & result) {
    PyObject * value = required_dict_item(dictionary, key);
    if (value == nullptr) {
        return false;
    }
    if (!PyLong_Check(value) || PyBool_Check(value)) {
        PyErr_Format(
            PyExc_TypeError,
            "Qwen35Runtime execution-plan field '%s' must be a non-negative int32",
            key);
        return false;
    }
    const long long parsed = PyLong_AsLongLong(value);
    if (PyErr_Occurred() || parsed < 0 || parsed > std::numeric_limits<std::int32_t>::max()) {
        PyErr_Clear();
        PyErr_Format(
            PyExc_ValueError,
            "Qwen35Runtime execution-plan field '%s' is outside the non-negative int32 range",
            key);
        return false;
    }
    result = static_cast<std::int32_t>(parsed);
    return true;
}

bool copy_int32_field(
    PyObject * dictionary,
    const char * key,
    std::size_t expected_size,
    std::vector<std::int32_t> & result) {
    PyObject * value = required_dict_item(dictionary, key);
    if (value == nullptr) {
        return false;
    }
    BufferView buffer;
    if (!buffer.acquire_int32_vector(value, key)) {
        return false;
    }
    if (buffer.size() != expected_size) {
        PyErr_Format(
            PyExc_ValueError,
            "Qwen35Runtime execution-plan field '%s' has %zu elements; expected %zu",
            key,
            buffer.size(),
            expected_size);
        return false;
    }

    result.resize(expected_size);
    if (expected_size != 0) {
        std::memcpy(result.data(), buffer.data(), expected_size * sizeof(std::int32_t));
    }
    return true;
}

bool parse_execution_plan(PyObject * object, Qwen35ExecutionPlan & plan) {
    if (!PyDict_CheckExact(object)) {
        PyErr_SetString(PyExc_TypeError, "Qwen35Runtime execution plan must be a dict");
        return false;
    }

    PyObject * is_prefill = required_dict_item(object, "is_prefill");
    if (is_prefill == nullptr || !parse_bool(is_prefill, "is_prefill", plan.is_prefill) ||
        !parse_plan_int32(object, "n_tokens", plan.n_tokens) ||
        !parse_plan_int32(object, "n_seqs", plan.n_seqs) ||
        !parse_plan_int32(object, "block_size", plan.block_size) ||
        !parse_plan_int32(object, "block_table_cols", plan.block_table_cols)) {
        return false;
    }

    const auto n_tokens = static_cast<std::size_t>(plan.n_tokens);
    const auto n_seqs = static_cast<std::size_t>(plan.n_seqs);
    const auto block_cols = static_cast<std::size_t>(plan.block_table_cols);
    if (n_seqs != 0 && block_cols > std::numeric_limits<std::size_t>::max() / n_seqs) {
        PyErr_SetString(PyExc_OverflowError, "Qwen35Runtime block-table dimensions overflow size_t");
        return false;
    }
    const std::size_t block_table_size = n_seqs * block_cols;

    if (!copy_int32_field(object, "tokens", n_tokens, plan.tokens) ||
        !copy_int32_field(object, "positions", n_tokens, plan.positions) ||
        !copy_int32_field(object, "seq_ids", n_seqs, plan.seq_ids) ||
        !copy_int32_field(
            object, "scheduled_token_counts", n_seqs, plan.scheduled_token_counts) ||
        !copy_int32_field(object, "slot_mapping", n_tokens, plan.slot_mapping) ||
        !copy_int32_field(object, "block_tables", block_table_size, plan.block_tables) ||
        !copy_int32_field(object, "context_lens", n_seqs, plan.context_lens) ||
        !copy_int32_field(
            object, "num_cached_tokens", n_seqs, plan.num_cached_tokens)) {
        return false;
    }

    return true;
}

PyObject * int32_list(const std::vector<std::int32_t> & values) {
    if (values.size() > static_cast<std::size_t>(PY_SSIZE_T_MAX)) {
        PyErr_SetString(PyExc_OverflowError, "Qwen35Runtime output is too large for a Python list");
        return nullptr;
    }
    PyObject * result = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        PyObject * value = PyLong_FromLong(values[index]);
        if (value == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(index), value);  // steals
    }
    return result;
}

Qwen35Runtime * require_runtime(PyQwen35Runtime * self) {
    if (self->runtime == nullptr || self->runtime->is_shutdown()) {
        PyErr_SetString(PyExc_RuntimeError, "Qwen35Runtime has been shut down");
        return nullptr;
    }
    return self->runtime;
}

int runtime_init(PyObject * self_object, PyObject * args, PyObject * kwargs) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    PyObject * model_path_object = nullptr;
    PyObject * backend_object = nullptr;
    PyObject * max_model_len_object = nullptr;
    PyObject * max_num_batched_tokens_object = nullptr;
    PyObject * max_num_seqs_object = nullptr;
    PyObject * block_size_object = nullptr;
    PyObject * num_blocks_object = nullptr;
    PyObject * n_threads_object = nullptr;
    PyObject * device_index_object = nullptr;
    PyObject * enable_mtp_object = nullptr;
    PyObject * mtp_max_draft_tokens_object = nullptr;
    PyObject * enable_graph_reuse_object = Py_True;
    PyObject * enable_vulkan_graph_reuse_object = Py_False;
    PyObject * attention_impl_object = nullptr;
    PyObject * enable_batched_recurrent_snapshots_object = Py_False;
    PyObject * enable_mtp_prefill_fusion_object = Py_False;
    PyObject * enable_mtp_verification_kv_fusion_object = Py_False;
    PyObject * mali_mtp_prefill_strategy_object = nullptr;
    static const char * keywords[] = {
        "model_path",
        "backend",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "block_size",
        "num_blocks",
        "n_threads",
        "device_index",
        "enable_mtp",
        "mtp_max_draft_tokens",
        "enable_graph_reuse",
        "enable_vulkan_graph_reuse",
        "attention_impl",
        "enable_batched_recurrent_snapshots",
        "enable_mtp_prefill_fusion",
        "enable_mtp_verification_kv_fusion",
        "mali_mtp_prefill_strategy",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOOOOOO|OOOOOOO:Qwen35Runtime",
            const_cast<char **>(keywords),
            &model_path_object,
            &backend_object,
            &max_model_len_object,
            &max_num_batched_tokens_object,
            &max_num_seqs_object,
            &block_size_object,
            &num_blocks_object,
            &n_threads_object,
            &device_index_object,
            &enable_mtp_object,
            &mtp_max_draft_tokens_object,
            &enable_graph_reuse_object,
            &enable_vulkan_graph_reuse_object,
            &attention_impl_object,
            &enable_batched_recurrent_snapshots_object,
            &enable_mtp_prefill_fusion_object,
            &enable_mtp_verification_kv_fusion_object,
            &mali_mtp_prefill_strategy_object)) {
        return -1;
    }
    if (self->runtime != nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "Qwen35Runtime is already initialized");
        return -1;
    }

    PyObject * path_bytes = nullptr;
    if (!PyUnicode_FSConverter(model_path_object, &path_bytes)) {
        return -1;
    }
    char * model_path = nullptr;
    Py_ssize_t model_path_size = 0;
    if (PyBytes_AsStringAndSize(path_bytes, &model_path, &model_path_size) < 0) {
        Py_DECREF(path_bytes);
        return -1;
    }
    if (!PyUnicode_Check(backend_object)) {
        Py_DECREF(path_bytes);
        PyErr_SetString(PyExc_TypeError, "Qwen35Runtime argument 'backend' must be str");
        return -1;
    }
    const char * backend = PyUnicode_AsUTF8(backend_object);
    if (backend == nullptr) {
        Py_DECREF(path_bytes);
        return -1;
    }

    Qwen35RuntimeOptions options;
    options.model_path.assign(model_path, static_cast<std::size_t>(model_path_size));
    options.backend = backend;
    Py_DECREF(path_bytes);
    if (options.model_path.empty() || options.model_path.find('\0') != std::string::npos) {
        PyErr_SetString(
            PyExc_ValueError,
            "Qwen35Runtime argument 'model_path' must be non-empty and contain no null bytes");
        return -1;
    }
    if (!parse_positive_size(max_model_len_object, "max_model_len", options.max_model_len) ||
        !parse_positive_size(
            max_num_batched_tokens_object,
            "max_num_batched_tokens",
            options.max_num_batched_tokens) ||
        !parse_positive_size(max_num_seqs_object, "max_num_seqs", options.max_num_seqs) ||
        !parse_positive_size(block_size_object, "block_size", options.block_size) ||
        !parse_positive_size(num_blocks_object, "num_blocks", options.num_blocks) ||
        !parse_positive_int(n_threads_object, "n_threads", options.cpu_threads) ||
        !parse_size(device_index_object, "device_index", options.device_index) ||
        !parse_bool(enable_mtp_object, "enable_mtp", options.enable_mtp) ||
        !parse_size(
            mtp_max_draft_tokens_object,
            "mtp_max_draft_tokens",
            options.mtp_max_draft_tokens) ||
        !parse_bool(
            enable_graph_reuse_object,
            "enable_graph_reuse",
            options.enable_graph_reuse) ||
        !parse_bool(
            enable_vulkan_graph_reuse_object,
            "enable_vulkan_graph_reuse",
            options.enable_vulkan_graph_reuse) ||
        !parse_bool(
            enable_batched_recurrent_snapshots_object,
            "enable_batched_recurrent_snapshots",
            options.enable_batched_recurrent_snapshots) ||
        !parse_bool(
            enable_mtp_prefill_fusion_object,
            "enable_mtp_prefill_fusion", options.enable_mtp_prefill_fusion) ||
        !parse_bool(
            enable_mtp_verification_kv_fusion_object,
            "enable_mtp_verification_kv_fusion",
            options.enable_mtp_verification_kv_fusion)) {
        return -1;
    }
    if (attention_impl_object != nullptr &&
        !parse_attention_implementation(
            attention_impl_object,
            options.attention_implementation)) {
        return -1;
    }
    if (mali_mtp_prefill_strategy_object != nullptr &&
        !parse_mali_mtp_prefill_strategy(
            mali_mtp_prefill_strategy_object,
            options.mali_mtp_prefill_strategy)) {
        return -1;
    }

    try {
        std::unique_ptr<Qwen35Runtime> runtime;
        {
            AllowThreads allow_threads;
            runtime = std::make_unique<Qwen35Runtime>(std::move(options));
        }
        self->runtime = runtime.release();
        return 0;
    } catch (...) {
        translate_cpp_exception();
        return -1;
    }
}

void runtime_dealloc(PyObject * self_object) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    // Destructors in the native runtime are noexcept by contract.  Releasing
    // the GIL avoids blocking unrelated Python threads on backend teardown.
    if (self->runtime != nullptr) {
        {
            AllowThreads allow_threads;
            delete self->runtime;
        }
        self->runtime = nullptr;
    }
    Py_TYPE(self_object)->tp_free(self_object);
}

PyObject * runtime_run(PyObject * self_object, PyObject * plan_object) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    Qwen35Runtime * runtime = require_runtime(self);
    if (runtime == nullptr) {
        return nullptr;
    }
    Qwen35ExecutionPlan plan;
    if (!parse_execution_plan(plan_object, plan)) {
        return nullptr;
    }

    try {
        std::vector<std::int32_t> token_ids;
        {
            AllowThreads allow_threads;
            token_ids = runtime->run(plan);
        }
        return int32_list(token_ids);
    } catch (...) {
        translate_cpp_exception();
        return nullptr;
    }
}

PyObject * runtime_run_mtp(PyObject * self_object, PyObject * args) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    PyObject * plan_object = nullptr;
    PyObject * capacity_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO:run_mtp", &plan_object, &capacity_object)) {
        return nullptr;
    }
    Qwen35Runtime * runtime = require_runtime(self);
    if (runtime == nullptr) {
        return nullptr;
    }
    Qwen35ExecutionPlan plan;
    std::size_t capacity = 0;
    if (!parse_execution_plan(plan_object, plan) ||
        !parse_positive_size(capacity_object, "capacity", capacity)) {
        return nullptr;
    }

    try {
        Qwen35MtpResult result;
        {
            AllowThreads allow_threads;
            result = runtime->run_mtp(plan, capacity);
        }
        PyObject * token_ids = int32_list(result.token_ids);
        if (token_ids == nullptr) {
            return nullptr;
        }
        PyObject * output_counts = int32_list(result.output_counts);
        if (output_counts == nullptr) {
            Py_DECREF(token_ids);
            return nullptr;
        }
        PyObject * draft_counts = int32_list(result.draft_counts);
        if (draft_counts == nullptr) {
            Py_DECREF(token_ids);
            Py_DECREF(output_counts);
            return nullptr;
        }
        PyObject * tuple = PyTuple_New(3);
        if (tuple == nullptr) {
            Py_DECREF(token_ids);
            Py_DECREF(output_counts);
            Py_DECREF(draft_counts);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, 0, token_ids);       // steals
        PyTuple_SET_ITEM(tuple, 1, output_counts);  // steals
        PyTuple_SET_ITEM(tuple, 2, draft_counts);   // steals
        return tuple;
    } catch (...) {
        translate_cpp_exception();
        return nullptr;
    }
}

PyObject * runtime_release_blocks(
    PyObject * self_object,
    PyObject * args,
    PyObject * kwargs) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    PyObject * block_ids_object = nullptr;
    PyObject * seq_ids_object = nullptr;
    PyObject * block_size_object = nullptr;
    static const char * keywords[] = {"block_ids", "seq_ids", "block_size", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOO:release_blocks",
            const_cast<char **>(keywords),
            &block_ids_object,
            &seq_ids_object,
            &block_size_object)) {
        return nullptr;
    }
    Qwen35Runtime * runtime = require_runtime(self);
    if (runtime == nullptr) {
        return nullptr;
    }

    BufferView block_ids_buffer;
    BufferView seq_ids_buffer;
    if (!block_ids_buffer.acquire_int32_vector(block_ids_object, "block_ids") ||
        !seq_ids_buffer.acquire_int32_vector(seq_ids_object, "seq_ids")) {
        return nullptr;
    }

    std::vector<std::int32_t> block_ids(block_ids_buffer.size());
    std::vector<std::int32_t> sequence_ids(seq_ids_buffer.size());
    if (!block_ids.empty()) {
        std::memcpy(
            block_ids.data(), block_ids_buffer.data(), block_ids.size() * sizeof(std::int32_t));
    }
    if (!sequence_ids.empty()) {
        std::memcpy(
            sequence_ids.data(), seq_ids_buffer.data(), sequence_ids.size() * sizeof(std::int32_t));
    }
    std::size_t block_size = 0;
    if (!parse_positive_size(block_size_object, "block_size", block_size)) {
        return nullptr;
    }

    try {
        {
            AllowThreads allow_threads;
            runtime->release_blocks(block_ids, sequence_ids, block_size);
        }
        Py_RETURN_NONE;
    } catch (...) {
        translate_cpp_exception();
        return nullptr;
    }
}

PyObject * runtime_shutdown(PyObject * self_object, PyObject *) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    if (self->runtime == nullptr || self->runtime->is_shutdown()) {
        Py_RETURN_NONE;
    }
    try {
        {
            AllowThreads allow_threads;
            self->runtime->shutdown();
        }
        Py_RETURN_NONE;
    } catch (...) {
        translate_cpp_exception();
        return nullptr;
    }
}

PyObject * runtime_graph_reuse_stats(PyObject * self_object, PyObject *) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    Qwen35Runtime * runtime = require_runtime(self);
    if (runtime == nullptr) {
        return nullptr;
    }
    try {
        Qwen35GraphReuseStats stats;
        {
            AllowThreads allow_threads;
            stats = runtime->graph_reuse_stats();
        }
        PyObject * result = PyDict_New();
        if (result == nullptr) {
            return nullptr;
        }
        const auto set_uint = [&](const char * key, std::uint64_t value) -> bool {
            PyObject * object = PyLong_FromUnsignedLongLong(
                static_cast<unsigned long long>(value));
            if (object == nullptr) {
                return false;
            }
            const int status = PyDict_SetItemString(result, key, object);
            Py_DECREF(object);
            return status == 0;
        };
        if (!set_uint("hits", stats.hits) ||
            !set_uint("misses", stats.misses) ||
            !set_uint("evictions", stats.evictions) ||
            !set_uint("active_entries", stats.active_entries)) {
            Py_DECREF(result);
            return nullptr;
        }
        return result;
    } catch (...) {
        translate_cpp_exception();
        return nullptr;
    }
}

PyObject * runtime_execution_profile_stats(PyObject * self_object, PyObject *) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    Qwen35Runtime * runtime = require_runtime(self);
    if (runtime == nullptr) {
        return nullptr;
    }
    try {
        Qwen35ExecutionProfileStats stats;
        {
            AllowThreads allow_threads;
            stats = runtime->execution_profile_stats();
        }
        PyObject * result = PyDict_New();
        if (result == nullptr) {
            return nullptr;
        }
        const auto set_string = [&](const char * key, const std::string & value) -> bool {
            PyObject * object = PyUnicode_FromStringAndSize(
                value.data(), static_cast<Py_ssize_t>(value.size()));
            if (object == nullptr) {
                return false;
            }
            const int status = PyDict_SetItemString(result, key, object);
            Py_DECREF(object);
            return status == 0;
        };
        const auto set_uint = [&](const char * key, std::uint64_t value) -> bool {
            PyObject * object = PyLong_FromUnsignedLongLong(
                static_cast<unsigned long long>(value));
            if (object == nullptr) {
                return false;
            }
            const int status = PyDict_SetItemString(result, key, object);
            Py_DECREF(object);
            return status == 0;
        };
        if (!set_string("backend", stats.backend) ||
            !set_string("device_name", stats.device_name) ||
            !set_string("device_description", stats.device_description) ||
            !set_string("profile_name", stats.profile_name) ||
            !set_string(
                "mali_mtp_prefill_strategy", stats.mali_mtp_prefill_strategy) ||
            !set_uint(
                "normal_prefill_chunk_tokens", stats.normal_prefill_chunk_tokens) ||
            !set_uint(
                "mtp_prefill_chunk_tokens", stats.mtp_prefill_chunk_tokens) ||
            !set_uint("staged_recurrent_planes", stats.staged_recurrent_planes)) {
            Py_DECREF(result);
            return nullptr;
        }
        return result;
    } catch (...) {
        translate_cpp_exception();
        return nullptr;
    }
}

PyObject * runtime_memory_stats(PyObject * self_object, PyObject *) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    Qwen35Runtime * runtime = require_runtime(self);
    if (runtime == nullptr) {
        return nullptr;
    }
    try {
        Qwen35MemoryStats stats;
        {
            AllowThreads allow_threads;
            stats = runtime->memory_stats();
        }
        PyObject * result = PyDict_New();
        if (result == nullptr) {
            return nullptr;
        }
        const auto set_uint = [&](const char * key, std::uint64_t value) -> bool {
            PyObject * object = PyLong_FromUnsignedLongLong(
                static_cast<unsigned long long>(value));
            if (object == nullptr) {
                return false;
            }
            const int status = PyDict_SetItemString(result, key, object);
            Py_DECREF(object);
            return status == 0;
        };
        if (!set_uint("weights_bytes", stats.weights_bytes) ||
            !set_uint("paged_kv_bytes", stats.paged_kv_bytes) ||
            !set_uint("recurrent_state_bytes", stats.recurrent_state_bytes) ||
            !set_uint("graph_metadata_bytes", stats.graph_metadata_bytes) ||
            !set_uint("graph_cache_entries", stats.graph_cache_entries) ||
            !set_uint("known_persistent_bytes", stats.known_persistent_bytes)) {
            Py_DECREF(result);
            return nullptr;
        }
        return result;
    } catch (...) {
        translate_cpp_exception();
        return nullptr;
    }
}

PyObject * runtime_mtp_profile_stats(PyObject * self_object, PyObject *) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(self_object);
    Qwen35Runtime * runtime = require_runtime(self);
    if (runtime == nullptr) {
        return nullptr;
    }
    try {
        Qwen35MtpProfileStats stats;
        {
            AllowThreads allow_threads;
            stats = runtime->mtp_profile_stats();
        }
        PyObject * result = PyDict_New();
        if (result == nullptr) {
            return nullptr;
        }
        const auto set_uint = [&](const char * key, std::uint64_t value) -> bool {
            PyObject * object = PyLong_FromUnsignedLongLong(
                static_cast<unsigned long long>(value));
            if (object == nullptr) {
                return false;
            }
            const int status = PyDict_SetItemString(result, key, object);
            Py_DECREF(object);
            return status == 0;
        };
        if (!set_uint("draft_calls", stats.draft_calls) ||
            !set_uint("draft_tokens", stats.draft_tokens) ||
            !set_uint("draft_graph_setup_elapsed_ns", stats.draft_graph_setup_elapsed_ns) ||
            !set_uint("draft_elapsed_ns", stats.draft_elapsed_ns) ||
            !set_uint("verification_calls", stats.verification_calls) ||
            !set_uint("verification_tokens", stats.verification_tokens) ||
            !set_uint("verification_graph_setup_elapsed_ns", stats.verification_graph_setup_elapsed_ns) ||
            !set_uint("verification_elapsed_ns", stats.verification_elapsed_ns) ||
            !set_uint("kv_update_calls", stats.kv_update_calls) ||
            !set_uint("kv_update_tokens", stats.kv_update_tokens) ||
            !set_uint("kv_update_graph_setup_elapsed_ns", stats.kv_update_graph_setup_elapsed_ns) ||
            !set_uint("kv_update_elapsed_ns", stats.kv_update_elapsed_ns)) {
            Py_DECREF(result);
            return nullptr;
        }
        return result;
    } catch (...) {
        translate_cpp_exception();
        return nullptr;
    }
}

PyCFunction cast_keyword_function(PyCFunctionWithKeywords function) noexcept {
    static_assert(
        sizeof(PyCFunction) == sizeof(PyCFunctionWithKeywords),
        "CPython method function pointer representations differ");
    PyCFunction result = nullptr;
    std::memcpy(&result, &function, sizeof(result));
    return result;
}

PyMethodDef runtime_methods[] = {
    {"run", runtime_run, METH_O,
     "Run one native greedy target-model execution plan."},
    {"run_mtp", runtime_run_mtp, METH_VARARGS,
     "Run native MTP draft, target verification, acceptance and rollback."},
    {"release_blocks", cast_keyword_function(runtime_release_blocks),
     METH_VARARGS | METH_KEYWORDS,
     "Release nano-vLLM Paged-KV blocks and native sequence state."},
    {"graph_reuse_stats", runtime_graph_reuse_stats, METH_NOARGS,
     "Return persistent graph cache hit/miss/eviction counters."},
    {"mtp_profile_stats", runtime_mtp_profile_stats, METH_NOARGS,
     "Return cumulative native MTP stage timing counters."},
    {"execution_profile_stats", runtime_execution_profile_stats, METH_NOARGS,
     "Return the effective native device execution policy."},
    {"memory_stats", runtime_memory_stats, METH_NOARGS,
     "Return native persistent-memory accounting by allocation class."},
    {"shutdown", runtime_shutdown, METH_NOARGS,
     "Synchronize and shut down the native runtime. This operation is idempotent."},
    {nullptr, nullptr, 0, nullptr},
};

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
PyTypeObject runtime_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

PyObject * runtime_new(PyTypeObject * type, PyObject *, PyObject *) {
    auto * self = reinterpret_cast<PyQwen35Runtime *>(type->tp_alloc(type, 0));
    if (self != nullptr) {
        self->runtime = nullptr;
    }
    return reinterpret_cast<PyObject *>(self);
}

}  // namespace

int register_qwen35_runtime_type(PyObject * module) {//注册Qwen35Runtime类型到Python模块
    if (module == nullptr) {
        PyErr_SetString(PyExc_SystemError, "cannot register Qwen35Runtime on a null module");
        return -1;
    }

    runtime_type.tp_name = "nanovllm._C.Qwen35Runtime";
    runtime_type.tp_basicsize = sizeof(PyQwen35Runtime);
    runtime_type.tp_itemsize = 0;
    runtime_type.tp_dealloc = runtime_dealloc;
    runtime_type.tp_flags = Py_TPFLAGS_DEFAULT;
    runtime_type.tp_doc =
        "In-tree Qwen3.5 GGUF runtime backed directly by embedded GGML CPU/Vulkan.";
    runtime_type.tp_methods = runtime_methods;
    runtime_type.tp_init = runtime_init;
    runtime_type.tp_new = runtime_new;

    if (PyType_Ready(&runtime_type) < 0) {
        return -1;
    }
    Py_INCREF(&runtime_type);
    if (PyModule_AddObject(
            module,
            "Qwen35Runtime",
            reinterpret_cast<PyObject *>(&runtime_type)) < 0) {
        Py_DECREF(&runtime_type);
        return -1;
    }
    return 0;
}

}  // namespace nanovllm::python
