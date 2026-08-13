#include "runtime/graph_executor.h"

#include "ggml.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace nanovllm::native {
namespace {

constexpr bool pure_metadata_view_op(enum ggml_op op) noexcept {
    return op == GGML_OP_VIEW ||
           op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

static_assert(pure_metadata_view_op(GGML_OP_VIEW));
static_assert(pure_metadata_view_op(GGML_OP_RESHAPE));
static_assert(pure_metadata_view_op(GGML_OP_PERMUTE));
static_assert(pure_metadata_view_op(GGML_OP_TRANSPOSE));
static_assert(!pure_metadata_view_op(GGML_OP_CPY));
static_assert(!pure_metadata_view_op(GGML_OP_SET_ROWS));

std::string safe_string(const char * value, const char * fallback) {
    return value != nullptr && value[0] != '\0' ? std::string(value) : std::string(fallback);
}

std::string tensor_label(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "<null tensor>";
    }
    if (tensor->name[0] != '\0') {
        return std::string(tensor->name);
    }
    return "<unnamed tensor>";
}

std::string status_label(ggml_status status) {
    const char * label = ggml_status_to_string(status);
    if (label != nullptr && label[0] != '\0') {
        return label;
    }
    return "status " + std::to_string(static_cast<int>(status));
}

void validate_backend_layout(const BackendList & backends) {
    if (backends.empty()) {
        throw GraphExecutorError("cannot create a graph executor with no backends");
    }

    if (backends.size() == 1 && backends.at(0).kind() == BackendKind::Cpu) {
        return;
    }
    if (backends.size() == 2 &&
        (backends.at(0).kind() == BackendKind::Vulkan ||
         backends.at(0).kind() == BackendKind::Cuda) &&
        backends.at(1).kind() == BackendKind::Cpu) {
        return;
    }

    std::ostringstream message;
    message << "unsupported graph executor backend layout [";
    for (std::size_t i = 0; i < backends.size(); ++i) {
        if (i != 0) {
            message << ", ";
        }
        message << backend_kind_name(backends.at(i).kind());
    }
    message << "]; expected [cpu], [vulkan, cpu], or [cuda, cpu]";
    throw GraphExecutorError(message.str());
}

}  // namespace

bool is_pure_metadata_view_op(enum ggml_op op) noexcept {
    return pure_metadata_view_op(op);
}

bool is_graph_compute_op(enum ggml_op op) noexcept {
    return op != GGML_OP_NONE && !pure_metadata_view_op(op);
}

GraphExecutorError::GraphExecutorError(const std::string & message)
    : std::runtime_error("native graph executor: " + message) {}

GraphExecutor::GraphExecutor(
    BackendList & backends,
    std::size_t graph_size,
    bool parallel,
    bool op_offload)
    : backends_(&backends), graph_size_(graph_size) {
    validate_backend_layout(backends);//检查后端布局是否符合要求目前只支持[cpu]或[vulkan, cpu]两种布局
    if (graph_size_ == 0) {
        throw GraphExecutorError("graph_size must be greater than zero");
    }

    backend_handles_ = backends.raw_handles();//获取后端的原始句柄列表
    buffer_types_ = backends.default_buffer_types();//获取后端的默认缓冲区类型列表
    if (backend_handles_.size() != buffer_types_.size()) {
        throw GraphExecutorError("backend handle and buffer-type counts do not match");
    }

    scheduler_ = ggml_backend_sched_new(
        backend_handles_.data(),
        buffer_types_.data(),
        static_cast<int>(backend_handles_.size()),
        graph_size_,
        parallel,
        op_offload);//创建一个新的GGML调度器，传入后端句柄、缓冲区类型、后端数量、图大小、是否并行和操作卸载标志
    if (scheduler_ == nullptr) {
        throw GraphExecutorError("ggml_backend_sched_new returned a null scheduler");
    }

    const int scheduler_backends = ggml_backend_sched_get_n_backends(scheduler_);// 获取调度器中后端的数量

    // 检查调度器是否保留了请求的后端数量，否则把调度器释放掉并抛出异常
    if (scheduler_backends != static_cast<int>(backend_handles_.size())) {
        ggml_backend_sched_free(scheduler_);        // 把调度器释放掉
        scheduler_ = nullptr;
        throw GraphExecutorError("GGML scheduler did not retain the requested backend count");
    }
    for (int i = 0; i < scheduler_backends; ++i) {
        if (ggml_backend_sched_get_backend(scheduler_, i) !=
            backend_handles_[static_cast<std::size_t>(i)]) {
            ggml_backend_sched_free(scheduler_);
            scheduler_ = nullptr;
            throw GraphExecutorError("GGML scheduler changed the requested backend order");
        }
    }
}

GraphExecutor::~GraphExecutor() {
    if (scheduler_ != nullptr) {
        ggml_backend_sched_synchronize(scheduler_);
        ggml_backend_sched_free(scheduler_);
    }
}

void GraphExecutor::require_graph(ggml_cgraph * graph, const char * operation) const {
    if (scheduler_ == nullptr) {
        throw GraphExecutorError(std::string(operation) + ": scheduler is null");
    }
    if (graph == nullptr) {
        throw GraphExecutorError(std::string(operation) + ": graph is null");
    }
    const int node_count = ggml_graph_n_nodes(graph);
    if (node_count < 0) {
        throw GraphExecutorError(std::string(operation) + ": graph has a negative node count");
    }
    if (static_cast<std::size_t>(node_count) > graph_size_) {
        std::ostringstream message;
        message << operation << ": graph contains " << node_count
                << " nodes, exceeding scheduler graph_size " << graph_size_;
        throw GraphExecutorError(message.str());
    }
}

void GraphExecutor::reset() {
    if (scheduler_ == nullptr) {
        throw GraphExecutorError("reset: scheduler is null");
    }
    ggml_backend_sched_synchronize(scheduler_);
    ggml_backend_sched_reset(scheduler_);
    allocated_graph_ = nullptr;
}

void GraphExecutor::reset_after_synchronous_compute() {
    if (scheduler_ == nullptr) {
        throw GraphExecutorError("reset after synchronous compute: scheduler is null");
    }
    ggml_backend_sched_reset(scheduler_);
    allocated_graph_ = nullptr;
}

void GraphExecutor::reserve(ggml_cgraph * measure_graph) {
    require_graph(measure_graph, "reserve");
    if (allocated_graph_ != nullptr) {
        throw GraphExecutorError(
            "reserve: a graph is currently allocated; call reset() and build a fresh measure graph first");
    }

    if (!ggml_backend_sched_reserve(scheduler_, measure_graph)) {
        ggml_backend_sched_synchronize(scheduler_);
        ggml_backend_sched_reset(scheduler_);
        throw GraphExecutorError(
            "reserve: GGML could not reserve transient buffers for the measure graph");
    }
}

void GraphExecutor::allocate(ggml_cgraph * graph) {
    require_graph(graph, "allocate");
    if (allocated_graph_ != nullptr) {
        if (allocated_graph_ == graph) {
            throw GraphExecutorError("allocate: this graph is already allocated");
        }
        throw GraphExecutorError(
            "allocate: another graph is allocated; call reset() before allocating a new graph");
    }

    if (!ggml_backend_sched_alloc_graph(scheduler_, graph)) {
        ggml_backend_sched_synchronize(scheduler_);
        ggml_backend_sched_reset(scheduler_);
        throw GraphExecutorError("allocate: GGML failed to allocate transient graph buffers");
    }
    allocated_graph_ = graph;
}

void GraphExecutor::compute(ggml_cgraph * graph) {
    require_graph(graph, "compute");
    if (allocated_graph_ == nullptr) {
        allocate(graph);
    } else if (allocated_graph_ != graph) {
        throw GraphExecutorError(
            "compute: the supplied graph differs from the allocated graph; call reset() first");
    }

    const ggml_status status = ggml_backend_sched_graph_compute(scheduler_, graph);
    if (status != GGML_STATUS_SUCCESS) {
        // The synchronous scheduler call has already synchronized its
        // backends.  Reset the failed allocation so it cannot be reused by
        // accident after an aborted/failed compute.
        ggml_backend_sched_reset(scheduler_);
        allocated_graph_ = nullptr;

        std::ostringstream message;
        message << "compute: GGML graph computation failed with " << status_label(status)
                << " (" << static_cast<int>(status) << ")";
        throw GraphExecutorError(message.str());
    }
}

void GraphExecutor::synchronize() {
    if (scheduler_ == nullptr) {
        throw GraphExecutorError("synchronize: scheduler is null");
    }
    ggml_backend_sched_synchronize(scheduler_);
}

std::size_t GraphExecutor::find_backend_index(BackendKind kind) const {
    for (std::size_t i = 0; i < backends_->size(); ++i) {
        if (backends_->at(i).kind() == kind) {
            return i;
        }
    }
    throw GraphExecutorError(
        std::string("backend '") + backend_kind_name(kind) + "' is not configured");
}

std::size_t GraphExecutor::find_backend_index(ggml_backend_t backend) const {
    if (backend == nullptr) {
        throw GraphExecutorError("backend handle is null");
    }
    const auto found = std::find(backend_handles_.begin(), backend_handles_.end(), backend);
    if (found == backend_handles_.end()) {
        throw GraphExecutorError("backend handle does not belong to this graph executor");
    }
    return static_cast<std::size_t>(std::distance(backend_handles_.begin(), found));
}

void GraphExecutor::set_tensor_backend(ggml_tensor * tensor, std::size_t backend_index) {
    if (tensor == nullptr) {
        throw GraphExecutorError("set tensor backend: tensor is null");
    }
    if (allocated_graph_ != nullptr) {
        throw GraphExecutorError(
            "set tensor backend: graph allocation already exists; call reset() and rebuild the graph first");
    }
    if (backend_index >= backend_handles_.size()) {
        std::ostringstream message;
        message << "set tensor backend: backend index " << backend_index
                << " is out of range for " << backend_handles_.size() << " backend(s)";
        throw GraphExecutorError(message.str());
    }

    ggml_backend_t backend = backend_handles_[backend_index];
    ggml_tensor * storage = resolve_view_source(tensor);
    if (storage->buffer != nullptr) {
        try {
            assert_buffer_on_backend(storage->buffer, backend, tensor_label(storage));
        } catch (const std::exception & error) {
            throw GraphExecutorError(
                "set tensor backend: incompatible preallocated storage for tensor '" +
                tensor_label(tensor) + "': " +
                std::string(error.what()));
        }
    }
    if (is_graph_compute_op(tensor->op) &&
        !ggml_backend_supports_op(backend, tensor)) {
        std::ostringstream message;
        message << "set tensor backend: backend '"
                << safe_string(ggml_backend_name(backend), "<unnamed>")
                << "' does not support " << safe_string(ggml_op_desc(tensor), "<unknown op>")
                << " for tensor '" << tensor_label(tensor) << "'";
        throw GraphExecutorError(message.str());
    }

    ggml_backend_sched_set_tensor_backend(scheduler_, tensor, backend);
}

void GraphExecutor::set_tensor_backend(ggml_tensor * tensor, BackendKind kind) {
    set_tensor_backend(tensor, find_backend_index(kind));
}

void GraphExecutor::set_tensor_backend(ggml_tensor * tensor, ggml_backend_t backend) {
    set_tensor_backend(tensor, find_backend_index(backend));
}

void GraphExecutor::set_all_compute_nodes_backend(
    ggml_cgraph * graph,
    BackendKind kind) {
    set_all_compute_nodes_backend(graph, backend_handles_[find_backend_index(kind)]);
}

void GraphExecutor::set_all_compute_nodes_backend(
    ggml_cgraph * graph,
    ggml_backend_t backend) {
    require_graph(graph, "set all compute nodes backend");
    if (allocated_graph_ != nullptr) {
        throw GraphExecutorError(
            "set all compute nodes backend: graph allocation already exists; "
            "call reset() and rebuild the graph first");
    }
    const std::size_t backend_index = find_backend_index(backend);
    const int node_count = ggml_graph_n_nodes(graph);
    for (int i = 0; i < node_count; ++i) {
        ggml_tensor * tensor = ggml_graph_node(graph, i);
        if (tensor == nullptr) {
            throw GraphExecutorError(
                "set all compute nodes backend: graph node " +
                std::to_string(i) + " is null");
        }
        if (!is_graph_compute_op(tensor->op)) {
            continue;
        }
        try {
            set_tensor_backend(tensor, backend_index);
        } catch (const GraphExecutorError & error) {
            std::ostringstream message;
            message << "set all compute nodes backend: graph node " << i
                    << " ('" << tensor_label(tensor) << "', "
                    << safe_string(ggml_op_desc(tensor), "<unknown op>")
                    << ") could not be assigned: " << error.what();
            throw GraphExecutorError(message.str());
        }
    }
}

ggml_tensor * GraphExecutor::resolve_view_source(ggml_tensor * tensor) const {
    if (tensor == nullptr) {
        throw GraphExecutorError("resolve view source: tensor is null");
    }

    std::unordered_set<const ggml_tensor *> visited;
    ggml_tensor * current = tensor;
    while (current->view_src != nullptr) {
        if (!visited.insert(current).second) {
            throw GraphExecutorError(
                "resolve view source: cycle detected at tensor '" + tensor_label(current) + "'");
        }
        current = current->view_src;
    }
    return current;
}

ggml_backend_t GraphExecutor::compute_backend(ggml_tensor * tensor) const {
    if (scheduler_ == nullptr) {
        throw GraphExecutorError("get compute backend: scheduler is null");
    }
    if (tensor == nullptr) {
        throw GraphExecutorError("get compute backend: tensor is null");
    }
    if (!is_graph_compute_op(tensor->op)) {
        return nullptr;
    }
    return ggml_backend_sched_get_tensor_backend(scheduler_, tensor);
}

ggml_backend_t GraphExecutor::storage_backend(ggml_tensor * tensor) const {
    if (scheduler_ == nullptr) {
        throw GraphExecutorError("get storage backend: scheduler is null");
    }
    ggml_tensor * storage = resolve_view_source(tensor);
    return ggml_backend_sched_get_tensor_backend(scheduler_, storage);
}

ggml_backend_t GraphExecutor::tensor_backend(ggml_tensor * tensor) const {
    return compute_backend(tensor);
}

void GraphExecutor::require_auditable_graph(ggml_cgraph * graph) const {
    require_graph(graph, "audit placement");
    if (allocated_graph_ == nullptr) {
        throw GraphExecutorError(
            "audit placement: graph placement is unavailable before allocate() or compute()");
    }
    if (allocated_graph_ != graph) {
        throw GraphExecutorError(
            "audit placement: supplied graph is not the currently allocated graph");
    }
}

GraphPlacementAudit GraphExecutor::audit_placement(ggml_cgraph * graph) const {
    require_auditable_graph(graph);

    GraphPlacementAudit audit;
    audit.split_count = ggml_backend_sched_get_n_splits(scheduler_);
    const int node_count = ggml_graph_n_nodes(graph);
    audit.nodes.reserve(static_cast<std::size_t>(node_count));

    for (int i = 0; i < node_count; ++i) {
        ggml_tensor * tensor = ggml_graph_node(graph, i);
        if (tensor == nullptr) {
            throw GraphExecutorError(
                "audit placement: graph node " + std::to_string(i) + " is null");
        }

        GraphNodePlacement placement;
        placement.node_index = static_cast<std::size_t>(i);
        placement.tensor = tensor;
        placement.storage_tensor = resolve_view_source(tensor);
        placement.tensor_name = tensor_label(tensor);
        placement.operation = safe_string(ggml_op_desc(tensor), "<unknown op>");
        placement.is_storage_view = placement.storage_tensor != tensor;
        placement.is_pure_metadata_view = is_pure_metadata_view_op(tensor->op);
        placement.is_compute = is_graph_compute_op(tensor->op);
        if (placement.is_storage_view) {
            ++audit.storage_view_nodes;
        }
        if (placement.is_pure_metadata_view) {
            ++audit.pure_metadata_view_nodes;
        }

        if (placement.is_compute) {
            const ggml_backend_t backend = compute_backend(tensor);
            if (backend == nullptr) {
                ++audit.unassigned_compute_nodes;
            } else {
                placement.compute.assigned = true;
                placement.compute.backend_index = find_backend_index(backend);
                placement.compute.backend_kind =
                    backends_->at(placement.compute.backend_index).kind();
                placement.compute.backend_name =
                    safe_string(ggml_backend_name(backend), "<unnamed>");
                switch (placement.compute.backend_kind) {
                    case BackendKind::Cpu:
                        ++audit.compute_cpu_nodes;
                        break;
                    case BackendKind::Vulkan:
                        ++audit.compute_vulkan_nodes;
                        break;
                    case BackendKind::Cuda:
                        ++audit.compute_cuda_nodes;
                        break;
                }
            }
        }

        const ggml_backend_t storage = storage_backend(tensor);
        if (storage == nullptr) {
            ++audit.unassigned_storage_nodes;
        } else {
            placement.storage.assigned = true;
            placement.storage.backend_index = find_backend_index(storage);
            placement.storage.backend_kind =
                backends_->at(placement.storage.backend_index).kind();
            placement.storage.backend_name =
                safe_string(ggml_backend_name(storage), "<unnamed>");
            switch (placement.storage.backend_kind) {
                case BackendKind::Cpu:
                    ++audit.storage_cpu_nodes;
                    break;
                case BackendKind::Vulkan:
                    ++audit.storage_vulkan_nodes;
                    break;
                case BackendKind::Cuda:
                    ++audit.storage_cuda_nodes;
                    break;
            }
        }
        audit.nodes.push_back(std::move(placement));
    }

    return audit;
}

void GraphExecutor::assert_all_compute_nodes_on_backend(
    ggml_cgraph * graph,
    BackendKind expected) const {
    const std::size_t expected_index = find_backend_index(expected);
    const GraphPlacementAudit audit = audit_placement(graph);
    std::size_t compute_nodes = 0;

    for (const GraphNodePlacement & placement : audit.nodes) {
        if (!placement.is_compute) {
            continue;
        }
        ++compute_nodes;
        if (!placement.compute.assigned) {
            throw GraphExecutorError(
                "placement assertion: compute tensor '" + placement.tensor_name +
                "' (" + placement.operation +
                ") has no scheduler compute-backend assignment");
        }
        if (placement.compute.backend_index == expected_index) {
            continue;
        }

        std::ostringstream message;
        message << "placement assertion: compute tensor '" << placement.tensor_name
                << "' (" << placement.operation << ") was assigned to "
                << backend_kind_name(placement.compute.backend_kind)
                << " backend '" << placement.compute.backend_name << "', expected "
                << backend_kind_name(expected) << " backend '"
                << backends_->at(expected_index).name() << "'";
        if (placement.is_storage_view) {
            message << "; its aliased storage backend is ";
            if (placement.storage.assigned) {
                message << backend_kind_name(placement.storage.backend_kind)
                        << " backend '" << placement.storage.backend_name << "'";
            } else {
                message << "unassigned";
            }
        }
        throw GraphExecutorError(message.str());
    }

    if (compute_nodes == 0) {
        throw GraphExecutorError("placement assertion: graph has no compute nodes");
    }
}

void GraphExecutor::assert_all_compute_nodes_on_vulkan(ggml_cgraph * graph) const {
    assert_all_compute_nodes_on_backend(graph, BackendKind::Vulkan);
}

void GraphExecutor::assert_critical_nodes_on_backend(
    ggml_cgraph * graph,
    const std::vector<ggml_tensor *> & critical_nodes,
    BackendKind expected) const {
    if (critical_nodes.empty()) {
        throw GraphExecutorError("placement assertion: critical node list is empty");
    }

    const std::size_t expected_index = find_backend_index(expected);
    const GraphPlacementAudit audit = audit_placement(graph);

    for (ggml_tensor * critical : critical_nodes) {
        if (critical == nullptr) {
            throw GraphExecutorError("placement assertion: critical node is null");
        }
        const auto found = std::find_if(
            audit.nodes.begin(),
            audit.nodes.end(),
            [critical](const GraphNodePlacement & placement) {
                return placement.tensor == critical;
            });
        if (found == audit.nodes.end()) {
            throw GraphExecutorError(
                "placement assertion: critical tensor '" + tensor_label(critical) +
                "' is not a graph node");
        }
        if (!found->is_compute) {
            throw GraphExecutorError(
                "placement assertion: critical tensor '" + found->tensor_name +
                "' (" + found->operation + ") is not a compute operation");
        }
        if (!found->compute.assigned) {
            throw GraphExecutorError(
                "placement assertion: critical tensor '" + found->tensor_name +
                "' (" + found->operation +
                ") has no scheduler compute-backend assignment");
        }
        if (found->compute.backend_index == expected_index) {
            continue;
        }

        std::ostringstream message;
        message << "placement assertion: critical compute tensor '"
                << found->tensor_name << "' (" << found->operation
                << ") was assigned to "
                << backend_kind_name(found->compute.backend_kind)
                << " backend '" << found->compute.backend_name << "', expected "
                << backend_kind_name(expected) << " backend '"
                << backends_->at(expected_index).name() << "'";
        if (found->is_storage_view) {
            message << "; this compute result aliases storage on ";
            if (found->storage.assigned) {
                message << backend_kind_name(found->storage.backend_kind)
                        << " backend '" << found->storage.backend_name << "'";
            } else {
                message << "an unassigned backend";
            }
        }
        throw GraphExecutorError(message.str());
    }
}

void GraphExecutor::assert_critical_nodes_on_vulkan(
    ggml_cgraph * graph,
    const std::vector<ggml_tensor *> & critical_nodes) const {
    assert_critical_nodes_on_backend(graph, critical_nodes, BackendKind::Vulkan);
}

}  // namespace nanovllm::native
