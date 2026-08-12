#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace nanovllm::native {

enum class BackendKind {
    Cpu,
    Vulkan,
    Cuda,
};

const char * backend_kind_name(BackendKind kind) noexcept;
BackendKind parse_backend_kind(std::string_view kind);
bool vulkan_backend_compiled() noexcept;
bool cuda_backend_compiled() noexcept;

struct BackendConfig {
    BackendKind kind = BackendKind::Cpu;
    int cpu_threads = 1;
    std::size_t device_index = 0;

    static BackendConfig cpu(int threads);
    static BackendConfig vulkan(std::size_t device_index = 0);
    static BackendConfig cuda(std::size_t device_index = 0);
};

struct BackendDeviceInfo {
    BackendKind kind = BackendKind::Cpu;
    std::size_t index = 0;
    std::string name;
    std::string description;
    std::size_t memory_free = 0;
    std::size_t memory_total = 0;
};

// Returns the CPU device followed by compiled and currently visible accelerator
// devices. Device discovery never creates a llama_model or llama_context.
std::vector<BackendDeviceInfo> available_backend_devices();

// Owns one GGML execution backend.  A moved-from Backend is valid only for
// destruction or assignment.
class Backend {
public:
    explicit Backend(const BackendConfig & config);
    ~Backend();

    Backend(const Backend &) = delete;
    Backend & operator=(const Backend &) = delete;

    Backend(Backend && other) noexcept;
    Backend & operator=(Backend && other) noexcept;

    ggml_backend_t get() const noexcept { return handle_; }
    BackendKind kind() const noexcept { return config_.kind; }
    const BackendConfig & config() const noexcept { return config_; }
    const BackendDeviceInfo & device_info() const noexcept { return device_info_; }
    bool has_persistent_threadpool() const noexcept { return threadpool_ != nullptr; }
    int threadpool_threads() const noexcept;

    const std::string & name() const noexcept { return name_; }
    ggml_backend_buffer_type_t default_buffer_type() const;
    bool supports_buffer_type(ggml_backend_buffer_type_t buffer_type) const noexcept;
    void synchronize() const;

private:
    void release() noexcept;

    ggml_backend_t handle_ = nullptr;
    ggml_threadpool_t threadpool_ = nullptr;
    BackendConfig config_{};
    BackendDeviceInfo device_info_{};
    std::string name_;
};

// Owns the ordered backends that a future ggml_backend_sched will consume.
// Accelerator execution is always [accelerator, cpu], because lower scheduler
// indices have higher priority. CPU-only execution is always [cpu]. This class does
// not create a scheduler or own graph-allocation buffers.
class BackendList {
public:
    static BackendList cpu_only(int cpu_threads);
    static BackendList vulkan_with_cpu(int cpu_threads, std::size_t vulkan_device_index = 0);
    static BackendList cuda_with_cpu(int cpu_threads, std::size_t cuda_device_index = 0);

    BackendList(const BackendList &) = delete;
    BackendList & operator=(const BackendList &) = delete;
    BackendList(BackendList &&) noexcept = default;
    BackendList & operator=(BackendList &&) noexcept = default;

    std::size_t size() const noexcept { return backends_.size(); }
    bool empty() const noexcept { return backends_.empty(); }
    Backend & at(std::size_t index);
    const Backend & at(std::size_t index) const;
    const std::vector<Backend> & entries() const noexcept { return backends_; }

    // The returned vectors retain BackendList's priority order and can be
    // passed directly to ggml_backend_sched_new by a later runtime layer.
    std::vector<ggml_backend_t> raw_handles() const;
    std::vector<ggml_backend_buffer_type_t> default_buffer_types() const;

private:
    BackendList() = default;
    std::vector<Backend> backends_;
};

std::string buffer_type_name(ggml_backend_buffer_type_t buffer_type);

bool backend_supports_buffer_type(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buffer_type) noexcept;

bool buffer_is_compatible_with_backend(
    ggml_backend_t backend,
    ggml_backend_buffer_t buffer) noexcept;

void assert_buffer_type_compatible(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buffer_type,
    std::string_view label = {});

void assert_buffer_on_backend(
    ggml_backend_buffer_t buffer,
    ggml_backend_t backend,
    std::string_view label = {});

// "On backend" means that the tensor has storage whose buffer type is
// supported by the backend.  Compatible shared/host buffer types are accepted.
void assert_tensor_on_backend(
    const ggml_tensor * tensor,
    ggml_backend_t backend,
    std::string_view label = {});

}  // namespace nanovllm::native
