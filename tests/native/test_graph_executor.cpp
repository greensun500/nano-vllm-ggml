#include "runtime/backend.h"
#include "runtime/graph_executor.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

void run_flash_attention_layout_test(nanovllm::native::BackendList & backends) {
    using nanovllm::native::BackendKind;
    using nanovllm::native::GraphExecutor;

    ContextHandle context;
    constexpr std::int64_t kHeadWidth = 2;
    constexpr std::int64_t kTokens = 1;
    constexpr std::int64_t kQueryHeads = 2;
    constexpr std::int64_t kKvHeads = 1;
    constexpr std::int64_t kKvTokens = 2;

    // Match the production graph layout exactly: the source is [D, H, T] and
    // FLASH_ATTN_EXT consumes the permuted [D, T, H] view.
    ggml_tensor * query_source = ggml_new_tensor_3d(
        context.get(), GGML_TYPE_F32, kHeadWidth, kQueryHeads, kTokens);
    ggml_tensor * key_source = ggml_new_tensor_3d(
        context.get(), GGML_TYPE_F32, kHeadWidth, kKvHeads, kKvTokens);
    ggml_tensor * value_source = ggml_new_tensor_3d(
        context.get(), GGML_TYPE_F32, kHeadWidth, kKvHeads, kKvTokens);
    ggml_tensor * mask = ggml_new_tensor_2d(
        context.get(), GGML_TYPE_F16, kKvTokens, kTokens);
    ggml_set_input(query_source);
    ggml_set_input(key_source);
    ggml_set_input(value_source);
    ggml_set_input(mask);

    ggml_tensor * query = ggml_permute(context.get(), query_source, 0, 2, 1, 3);
    ggml_tensor * key = ggml_permute(context.get(), key_source, 0, 2, 1, 3);
    ggml_tensor * value = ggml_permute(context.get(), value_source, 0, 2, 1, 3);
    ggml_tensor * attended = ggml_flash_attn_ext(
        context.get(), query, key, value, mask, 1.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    ggml_tensor * flattened = ggml_reshape_2d(
        context.get(), attended, kHeadWidth * kQueryHeads, kTokens);
    ggml_set_output(flattened);

    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 32, false);
    ggml_build_forward_expand(graph, flattened);
    GraphExecutor executor(backends, 32);
    executor.set_all_compute_nodes_backend(graph, BackendKind::Cpu);
    executor.allocate(graph);
    executor.assert_all_compute_nodes_on_backend(graph, BackendKind::Cpu);

    // Head 0 queries [1, 0]; head 1 queries [0, 1]. K rows are the two
    // basis vectors, and V rows make the expected GQA outputs easy to check.
    const std::array<float, 4> query_values{1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<float, 4> key_values{1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<float, 4> value_values{1.0f, 10.0f, 2.0f, 20.0f};
    const std::array<float, 2> mask_values{0.0f, 0.0f};
    std::array<ggml_fp16_t, 2> mask_values_f16{};
    ggml_fp32_to_fp16_row(
        mask_values.data(), mask_values_f16.data(), static_cast<std::int64_t>(mask_values.size()));
    ggml_backend_tensor_set(
        query_source, query_values.data(), 0, sizeof(query_values));
    ggml_backend_tensor_set(key_source, key_values.data(), 0, sizeof(key_values));
    ggml_backend_tensor_set(value_source, value_values.data(), 0, sizeof(value_values));
    ggml_backend_tensor_set(mask, mask_values_f16.data(), 0, sizeof(mask_values_f16));
    executor.compute(graph);

    std::array<float, 4> actual{};
    ggml_backend_tensor_get(flattened, actual.data(), 0, sizeof(actual));
    const float p = std::exp(1.0f) / (std::exp(1.0f) + 1.0f);
    const std::array<float, 4> expected{
        p * 1.0f + (1.0f - p) * 2.0f,
        p * 10.0f + (1.0f - p) * 20.0f,
        (1.0f - p) * 1.0f + p * 2.0f,
        (1.0f - p) * 10.0f + p * 20.0f,
    };
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(
            std::abs(actual[index] - expected[index]) < 1e-5f,
            "FLASH_ATTN_EXT result or flattened layout is incorrect");
    }

    const std::array<float, 2> masked_values{
        0.0f, -std::numeric_limits<float>::infinity()};
    ggml_fp32_to_fp16_row(
        masked_values.data(),
        mask_values_f16.data(),
        static_cast<std::int64_t>(masked_values.size()));
    ggml_backend_tensor_set(mask, mask_values_f16.data(), 0, sizeof(mask_values_f16));
    executor.compute(graph);
    ggml_backend_tensor_get(flattened, actual.data(), 0, sizeof(actual));
    const std::array<float, 4> masked_expected{1.0f, 10.0f, 1.0f, 10.0f};
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(
            std::abs(actual[index] - masked_expected[index]) < 1e-5f,
            "FLASH_ATTN_EXT did not honor its F16 causal mask");
    }
}

void run_contiguous_snapshot_copy_test(nanovllm::native::BackendList & backends) {
    using nanovllm::native::BackendKind;
    using nanovllm::native::GraphExecutor;

    constexpr std::int64_t kStateElements = 8;
    constexpr std::int64_t kSnapshotCount = 3;
    constexpr std::int64_t kOutputRowsBeforeSnapshots = 2;
    constexpr std::int64_t kTotalRows = 6;
    constexpr std::int64_t kAttentionOutputElements = 5;
    const std::size_t state_bytes = ggml_row_size(GGML_TYPE_F32, kStateElements);

    ContextHandle storage_context;
    ggml_tensor * storage = ggml_new_tensor_2d(
        storage_context.get(), GGML_TYPE_F32, kStateElements, kTotalRows);
    BufferHandle storage_buffer(
        ggml_backend_alloc_ctx_tensors(storage_context.get(), backends.at(0).get()));
    std::vector<float> initial_storage(
        static_cast<std::size_t>(kStateElements * kTotalRows), -1.0f);
    ggml_backend_tensor_set(
        storage, initial_storage.data(), 0, initial_storage.size() * sizeof(float));

    ContextHandle graph_context;
    const std::int64_t source_elements =
        kAttentionOutputElements + kStateElements * kSnapshotCount;
    ggml_tensor * gdn_output = ggml_new_tensor_1d(
        graph_context.get(), GGML_TYPE_F32, source_elements);
    ggml_set_input(gdn_output);
    ggml_tensor * snapshot_source = ggml_view_4d(
        graph_context.get(),
        gdn_output,
        kStateElements,
        1,
        kSnapshotCount,
        1,
        state_bytes,
        state_bytes,
        state_bytes * static_cast<std::size_t>(kSnapshotCount),
        static_cast<std::size_t>(kAttentionOutputElements) * sizeof(float));
    ggml_tensor * snapshot_destination = ggml_view_4d(
        graph_context.get(),
        storage,
        kStateElements,
        1,
        kSnapshotCount,
        1,
        state_bytes,
        state_bytes,
        state_bytes * static_cast<std::size_t>(kSnapshotCount),
        static_cast<std::size_t>(kOutputRowsBeforeSnapshots) * state_bytes);
    require(ggml_is_contiguous(snapshot_source), "GDN snapshot source must be contiguous");
    require(
        ggml_is_contiguous(snapshot_destination),
        "recurrent snapshot destination must be contiguous");

    ggml_tensor * update = ggml_cpy(
        graph_context.get(), snapshot_source, snapshot_destination);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context.get(), 32, false);
    ggml_build_forward_expand(graph, update);
    GraphExecutor executor(backends, 32);
    executor.set_all_compute_nodes_backend(graph, BackendKind::Cpu);
    executor.allocate(graph);

    std::vector<float> source(static_cast<std::size_t>(source_elements));
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(index);
    }
    ggml_backend_tensor_set(
        gdn_output, source.data(), 0, source.size() * sizeof(float));
    executor.compute(graph);

    std::vector<float> actual_storage(initial_storage.size());
    ggml_backend_tensor_get(
        storage,
        actual_storage.data(),
        0,
        actual_storage.size() * sizeof(float));
    for (std::int64_t plane = 0; plane < kSnapshotCount; ++plane) {
        for (std::int64_t element = 0; element < kStateElements; ++element) {
            const std::size_t source_index = static_cast<std::size_t>(
                kAttentionOutputElements + plane * kStateElements + element);
            const std::size_t destination_index = static_cast<std::size_t>(
                (kOutputRowsBeforeSnapshots + plane) * kStateElements + element);
            require(
                actual_storage[destination_index] == source[source_index],
                "batched snapshot copy changed plane ordering");
        }
    }
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

    run_flash_attention_layout_test(backends);
    run_contiguous_snapshot_copy_test(backends);

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
