#include "runtime/backend.h"
#include "runtime/graph_executor.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

class ContextHandle {
public:
    ContextHandle() {
        ggml_init_params params{};
        params.mem_size = 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        value_ = ggml_init(params);
        if (value_ == nullptr) {
            throw std::runtime_error("ggml_init failed");
        }
    }

    ~ContextHandle() {
        if (value_ != nullptr) {
            ggml_free(value_);
        }
    }

    ContextHandle(const ContextHandle &) = delete;
    ContextHandle & operator=(const ContextHandle &) = delete;

    ggml_context * get() const noexcept { return value_; }

private:
    ggml_context * value_ = nullptr;
};

class BufferHandle {
public:
    explicit BufferHandle(ggml_backend_buffer_t value) : value_(value) {
        if (value_ == nullptr) {
            throw std::runtime_error("backend buffer allocation failed");
        }
    }

    ~BufferHandle() {
        if (value_ != nullptr) {
            ggml_backend_buffer_free(value_);
        }
    }

    BufferHandle(const BufferHandle &) = delete;
    BufferHandle & operator=(const BufferHandle &) = delete;

private:
    ggml_backend_buffer_t value_ = nullptr;
};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_cpu(const nanovllm::native::BackendPlacement & placement, const char * label) {
    require(placement.assigned, std::string(label) + " is unassigned");
    require(
        placement.backend_kind == nanovllm::native::BackendKind::Cpu,
        std::string(label) + " is not on CPU");
}

void run_test() {
    using nanovllm::native::BackendKind;
    using nanovllm::native::BackendList;
    using nanovllm::native::GraphExecutor;

    require(nanovllm::native::is_pure_metadata_view_op(GGML_OP_VIEW),
            "VIEW must be a pure metadata view");
    require(nanovllm::native::is_pure_metadata_view_op(GGML_OP_RESHAPE),
            "RESHAPE must be a pure metadata view");
    require(!nanovllm::native::is_pure_metadata_view_op(GGML_OP_CPY),
            "CPY must not be a pure metadata view");
    require(!nanovllm::native::is_pure_metadata_view_op(GGML_OP_SET_ROWS),
            "SET_ROWS must not be a pure metadata view");

    BackendList backends = BackendList::cpu_only(1);
    require(backends.at(0).has_persistent_threadpool(),
            "CPU backend must own a persistent threadpool");
    require(backends.at(0).threadpool_threads() == 1,
            "CPU threadpool must use the configured thread count");

    nanovllm::native::Backend moved_from(nanovllm::native::BackendConfig::cpu(2));
    nanovllm::native::Backend moved_to(std::move(moved_from));
    require(moved_from.get() == nullptr && !moved_from.has_persistent_threadpool(),
            "moving a CPU backend must clear both source handles");
    require(moved_to.has_persistent_threadpool() && moved_to.threadpool_threads() == 2,
            "moving a CPU backend must transfer its persistent threadpool");
    nanovllm::native::Backend move_assigned(nanovllm::native::BackendConfig::cpu(1));
    move_assigned = std::move(moved_to);
    require(moved_to.get() == nullptr && !moved_to.has_persistent_threadpool(),
            "move-assigning a CPU backend must clear both source handles");
    require(move_assigned.has_persistent_threadpool() &&
                move_assigned.threadpool_threads() == 2,
            "move-assigning a CPU backend must transfer its persistent threadpool");

    ContextHandle persistent_context;
    ggml_tensor * copy_destination =
        ggml_new_tensor_1d(persistent_context.get(), GGML_TYPE_F32, 4);
    ggml_tensor * rows_destination =
        ggml_new_tensor_2d(persistent_context.get(), GGML_TYPE_F32, 4, 2);
    ggml_set_name(copy_destination, "test.copy_destination");
    ggml_set_name(rows_destination, "test.rows_destination");
    BufferHandle persistent_buffer(
        ggml_backend_alloc_ctx_tensors(persistent_context.get(), backends.at(0).get()));

    const std::array<float, 4> zeros4{};
    const std::array<float, 8> zeros8{};
    ggml_backend_tensor_set(copy_destination, zeros4.data(), 0, sizeof(zeros4));
    ggml_backend_tensor_set(rows_destination, zeros8.data(), 0, sizeof(zeros8));

    ContextHandle graph_context;
    ggml_tensor * copy_source =
        ggml_new_tensor_1d(graph_context.get(), GGML_TYPE_F32, 4);
    ggml_tensor * rows_source =
        ggml_new_tensor_2d(graph_context.get(), GGML_TYPE_F32, 4, 1);
    ggml_tensor * row_index =
        ggml_new_tensor_1d(graph_context.get(), GGML_TYPE_I32, 1);
    ggml_set_name(copy_source, "test.copy_source");
    ggml_set_name(rows_source, "test.rows_source");
    ggml_set_name(row_index, "test.row_index");
    ggml_set_input(copy_source);
    ggml_set_input(rows_source);
    ggml_set_input(row_index);

    ggml_tensor * reshaped = ggml_reshape_2d(graph_context.get(), copy_source, 2, 2);
    ggml_tensor * viewed = ggml_view_1d(graph_context.get(), reshaped, 4, 0);
    ggml_tensor * copied = ggml_cpy(graph_context.get(), viewed, copy_destination);
    ggml_tensor * rows_set =
        ggml_set_rows(graph_context.get(), rows_destination, rows_source, row_index);
    ggml_set_name(reshaped, "test.reshape");
    ggml_set_name(viewed, "test.view");
    ggml_set_name(copied, "test.copy");
    ggml_set_name(rows_set, "test.set_rows");
    ggml_set_output(copied);
    ggml_set_output(rows_set);

    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 32, false);
    ggml_build_forward_expand(graph, copied);
    ggml_build_forward_expand(graph, rows_set);

    GraphExecutor executor(backends, 32);
    executor.set_all_compute_nodes_backend(graph, BackendKind::Cpu);
    executor.allocate(graph);
    const nanovllm::native::GraphPlacementAudit audit = executor.audit_placement(graph);
    executor.assert_all_compute_nodes_on_backend(graph, BackendKind::Cpu);

    bool saw_copy = false;
    bool saw_set_rows = false;
    bool saw_reshape = false;
    bool saw_view = false;
    for (const nanovllm::native::GraphNodePlacement & node : audit.nodes) {
        if (node.tensor->op == GGML_OP_CPY || node.tensor->op == GGML_OP_SET_ROWS) {
            require(node.is_storage_view, node.operation + " must alias destination storage");
            require(node.is_compute, node.operation + " must be audited as compute");
            require(!node.is_pure_metadata_view,
                    node.operation + " must not be a pure metadata view");
            require_cpu(node.compute, (node.operation + " compute backend").c_str());
            require_cpu(node.storage, (node.operation + " storage backend").c_str());
            saw_copy = saw_copy || node.tensor->op == GGML_OP_CPY;
            saw_set_rows = saw_set_rows || node.tensor->op == GGML_OP_SET_ROWS;
        }
        if (node.tensor->op == GGML_OP_RESHAPE || node.tensor->op == GGML_OP_VIEW) {
            require(node.is_storage_view, node.operation + " must alias source storage");
            require(node.is_pure_metadata_view,
                    node.operation + " must be a pure metadata view");
            require(!node.is_compute, node.operation + " must not have compute placement");
            require(!node.compute.assigned,
                    node.operation + " must not report a compute backend");
            require_cpu(node.storage, (node.operation + " storage backend").c_str());
            saw_reshape = saw_reshape || node.tensor->op == GGML_OP_RESHAPE;
            saw_view = saw_view || node.tensor->op == GGML_OP_VIEW;
        }
    }
    require(saw_copy, "audit did not contain CPY");
    require(saw_set_rows, "audit did not contain SET_ROWS");
    require(saw_reshape, "audit did not contain RESHAPE");
    require(saw_view, "audit did not contain VIEW");
    require(audit.compute_cpu_nodes == 2, "expected exactly two CPU compute nodes");
    require(audit.compute_vulkan_nodes == 0, "unexpected Vulkan compute node");
    require(audit.unassigned_compute_nodes == 0, "unassigned compute node");
    require(audit.pure_metadata_view_nodes == 2,
            "expected exactly two pure metadata view nodes");

    const std::array<float, 4> copy_data{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> row_data{5.0f, 6.0f, 7.0f, 8.0f};
    const std::int32_t index = 1;
    ggml_backend_tensor_set(copy_source, copy_data.data(), 0, sizeof(copy_data));
    ggml_backend_tensor_set(rows_source, row_data.data(), 0, sizeof(row_data));
    ggml_backend_tensor_set(row_index, &index, 0, sizeof(index));
    // Re-enter graph execution through the same Backend to exercise reuse of
    // its externally owned threadpool across compute calls.
    executor.compute(graph);
    executor.compute(graph);
    executor.synchronize();

    std::array<float, 4> actual_copy{};
    std::array<float, 8> actual_rows{};
    ggml_backend_tensor_get(
        copy_destination, actual_copy.data(), 0, sizeof(actual_copy));
    ggml_backend_tensor_get(
        rows_destination, actual_rows.data(), 0, sizeof(actual_rows));
    require(actual_copy == copy_data, "CPY result is incorrect");
    require(actual_rows[0] == 0.0f && actual_rows[1] == 0.0f &&
                actual_rows[2] == 0.0f && actual_rows[3] == 0.0f,
            "SET_ROWS modified the wrong row");
    for (std::size_t i = 0; i < row_data.size(); ++i) {
        require(actual_rows[4 + i] == row_data[i], "SET_ROWS result is incorrect");
    }
}

}  // namespace

int main() {
    try {
        run_test();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "graph executor placement test failed: " << error.what() << '\n';
        return 1;
    }
}
