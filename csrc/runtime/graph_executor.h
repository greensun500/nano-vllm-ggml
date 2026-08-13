#pragma once

#include "runtime/backend.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanovllm::native {

class GraphExecutorError : public std::runtime_error {
public:
    explicit GraphExecutorError(const std::string & message);
};

// A backend selected for either computation or storage.  Computation and
// storage are intentionally separate: GGML_OP_CPY and GGML_OP_SET_ROWS are
// real compute nodes whose result aliases a destination tensor.
struct BackendPlacement {
    static constexpr std::size_t kUnassignedBackend =
        std::numeric_limits<std::size_t>::max();

    bool assigned = false;
    std::size_t backend_index = kUnassignedBackend;
    BackendKind backend_kind = BackendKind::Cpu;
    std::string backend_name;
};

// One scheduler decision for a graph node.  Pure metadata view operations do
// not have a compute placement; their storage placement still identifies the
// backend of the ultimate view source.
struct GraphNodePlacement {
    std::size_t node_index = 0;
    ggml_tensor * tensor = nullptr;
    ggml_tensor * storage_tensor = nullptr;
    std::string tensor_name;
    std::string operation;
    bool is_storage_view = false;
    bool is_pure_metadata_view = false;
    bool is_compute = false;
    BackendPlacement compute;
    BackendPlacement storage;
};

struct GraphPlacementAudit {
    std::vector<GraphNodePlacement> nodes;
    std::size_t compute_cpu_nodes = 0;
    std::size_t compute_vulkan_nodes = 0;
    std::size_t compute_cuda_nodes = 0;
    std::size_t unassigned_compute_nodes = 0;
    std::size_t storage_cpu_nodes = 0;
    std::size_t storage_vulkan_nodes = 0;
    std::size_t storage_cuda_nodes = 0;
    std::size_t unassigned_storage_nodes = 0;
    std::size_t storage_view_nodes = 0;
    std::size_t pure_metadata_view_nodes = 0;
    int split_count = 0;
};

// Keep this definition aligned with GGML scheduler's ggml_is_view_op().  A
// tensor merely having view_src is not enough: CPY and SET_ROWS have view
// storage but still execute kernels.
bool is_pure_metadata_view_op(enum ggml_op op) noexcept;
bool is_graph_compute_op(enum ggml_op op) noexcept;

// Owns only GGML's transient graph scheduler/allocation state.  BackendList,
// graph metadata, persistent model weights, KV cache, and recurrent-state
// buffers must all outlive the operations that use them and remain owned by
// their respective runtime components.
//
// Supported backend layouts are deliberately narrow:
//   * CPU:    [cpu]
//   * Vulkan: [vulkan, cpu]
//   * CUDA:   [cuda, cpu]
//
// The CPU entry in the Vulkan layout remains available for scheduler-managed
// graph inputs and transfers.  A strict Vulkan model graph should call
// set_all_compute_nodes_backend() before allocate(), then
// assert_all_compute_nodes_on_vulkan() after allocate(), so every real
// operation (including CPY and SET_ROWS) is verified rather than sampled.
class GraphExecutor {
public:
    explicit GraphExecutor(
        BackendList & backends,
        std::size_t graph_size = GGML_DEFAULT_GRAPH_SIZE,
        bool parallel = false,
        bool op_offload = true);
    ~GraphExecutor();

    GraphExecutor(const GraphExecutor &) = delete;
    GraphExecutor & operator=(const GraphExecutor &) = delete;
    GraphExecutor(GraphExecutor &&) = delete;
    GraphExecutor & operator=(GraphExecutor &&) = delete;

    BackendList & backends() noexcept { return *backends_; }
    const BackendList & backends() const noexcept { return *backends_; }

    std::size_t graph_size() const noexcept { return graph_size_; }
    bool has_allocation() const noexcept { return allocated_graph_ != nullptr; }
    ggml_cgraph * allocated_graph() const noexcept { return allocated_graph_; }

    // Synchronizes outstanding work before invalidating scheduler-owned graph
    // buffers.  Tensors allocated by the previous graph must be discarded.
    void reset();

    // `compute()` is synchronous in the GGML scheduler API.  Call this after
    // a successful compute when the graph will be discarded immediately; it
    // avoids issuing a second backend synchronization before the reset.  It
    // must not be used after allocation/build failures or async backend work.
    void reset_after_synchronous_compute();

    // Measures and reserves transient scheduler buffers.  The measure graph is
    // single-use according to the GGML scheduler contract.
    void reserve(ggml_cgraph * measure_graph);

    // Splits and allocates a graph without computing it.  This is the point at
    // which placement auditing becomes available.
    void allocate(ggml_cgraph * graph);
    void alloc(ggml_cgraph * graph) { allocate(graph); }

    // Synchronous computation.  If necessary, the graph is allocated first.
    // Repeated calls are allowed only for the same allocated graph.
    void compute(ggml_cgraph * graph);
    void synchronize();
    void sync() { synchronize(); }

    // Explicit assignments must be made before allocate()/compute().  The
    // backend must be one of this executor's scheduler backends.
    void set_tensor_backend(ggml_tensor * tensor, std::size_t backend_index);
    void set_tensor_backend(ggml_tensor * tensor, BackendKind kind);
    void set_tensor_backend(ggml_tensor * tensor, ggml_backend_t backend);

    // Assigns every non-NONE, non-metadata-view graph operation.  This must be
    // called before allocate()/compute(); unsupported operations fail here
    // instead of silently falling back during scheduler splitting.
    void set_all_compute_nodes_backend(ggml_cgraph * graph, BackendKind kind);
    void set_all_compute_nodes_backend(ggml_cgraph * graph, ggml_backend_t backend);

    // Direct scheduler assignment for the operation represented by tensor.
    // Pure metadata views have no compute placement even if GGML internally
    // records a backend id for storage propagation.
    ggml_backend_t compute_backend(ggml_tensor * tensor) const;

    // Scheduler assignment for tensor's ultimate view source (where bytes
    // live).  For CPY/SET_ROWS this can differ conceptually from compute_backend().
    ggml_backend_t storage_backend(ggml_tensor * tensor) const;

    // Backward-compatible spelling: tensor backend now means compute backend,
    // never implicit storage-root placement.
    ggml_backend_t tensor_backend(ggml_tensor * tensor) const;

    // Placement information is authoritative only after allocate() (or after
    // compute() auto-allocates).  An unassigned node is retained in the report
    // and can therefore never accidentally pass an assertion.
    GraphPlacementAudit audit_placement(ggml_cgraph * graph) const;

    void assert_all_compute_nodes_on_backend(
        ggml_cgraph * graph,
        BackendKind expected) const;

    void assert_all_compute_nodes_on_vulkan(ggml_cgraph * graph) const;

    // Compatibility helper for narrow diagnostics.  Strict model execution
    // should prefer assert_all_compute_nodes_on_backend().
    void assert_critical_nodes_on_backend(
        ggml_cgraph * graph,
        const std::vector<ggml_tensor *> & critical_nodes,
        BackendKind expected) const;

    void assert_critical_nodes_on_vulkan(
        ggml_cgraph * graph,
        const std::vector<ggml_tensor *> & critical_nodes) const;

private:
    std::size_t find_backend_index(BackendKind kind) const;
    std::size_t find_backend_index(ggml_backend_t backend) const;
    ggml_tensor * resolve_view_source(ggml_tensor * tensor) const;
    void require_graph(ggml_cgraph * graph, const char * operation) const;
    void require_auditable_graph(ggml_cgraph * graph) const;

    BackendList * backends_ = nullptr;  // non-owning

    // Keep stable arrays for the scheduler's complete lifetime.  Current GGML
    // copies their entries in ggml_backend_sched_new(); retaining them here
    // also avoids relying on pointers into temporary vectors if that API ever
    // changes within the pinned source revision.
    std::vector<ggml_backend_t> backend_handles_;
    std::vector<ggml_backend_buffer_type_t> buffer_types_;
    ggml_backend_sched_t scheduler_ = nullptr;

    std::size_t graph_size_ = 0;
    ggml_cgraph * allocated_graph_ = nullptr;  // non-owning
};

}  // namespace nanovllm::native
