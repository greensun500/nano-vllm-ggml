#include "runtime/backend.h"

#include "ggml-cpu.h"

#ifndef NANOVLLM_NATIVE_HAS_VULKAN
#define NANOVLLM_NATIVE_HAS_VULKAN 0
#endif

#if NANOVLLM_NATIVE_HAS_VULKAN
#include "ggml-vulkan.h"
#endif

#include <array>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace nanovllm::native {
namespace {

std::string safe_string(const char * value, std::string_view fallback = {}) {
    return value != nullptr && value[0] != '\0' ? std::string(value) : std::string(fallback);
}

std::string labelled_message(std::string_view label, std::string_view message) {
    if (label.empty()) {
        return std::string(message);
    }
    std::string result;
    result.reserve(label.size() + message.size() + 2);
    result.append(label);
    result.append(": ");
    result.append(message);
    return result;
}

void require_backend(ggml_backend_t backend, std::string_view operation) {
    if (backend == nullptr) {
        throw std::runtime_error(std::string(operation) + ": backend handle is null");
    }
}

BackendDeviceInfo describe_backend(
    ggml_backend_t backend,
    BackendKind kind,
    std::size_t index,
    std::string_view fallback_description = {}) {
    require_backend(backend, "describe backend");

    BackendDeviceInfo info;
    info.kind = kind;
    info.index = index;

    const ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device != nullptr) {
        info.name = safe_string(ggml_backend_dev_name(device), ggml_backend_name(backend));
        info.description = safe_string(ggml_backend_dev_description(device), fallback_description);
        ggml_backend_dev_memory(device, &info.memory_free, &info.memory_total);
    } else {
        info.name = safe_string(ggml_backend_name(backend), backend_kind_name(kind));
        info.description = std::string(fallback_description);
    }
    if (info.description.empty()) {
        info.description = info.name;
    }
    return info;
}

ggml_backend_t create_backend_handle(const BackendConfig & config) {
    switch (config.kind) {
        case BackendKind::Cpu: {
            if (config.cpu_threads <= 0) {
                throw std::runtime_error("CPU thread count must be greater than zero");
            }
            if (config.device_index != 0) {
                throw std::runtime_error("CPU backend only supports device index 0");
            }
            ggml_backend_t backend = ggml_backend_cpu_init();
            if (backend == nullptr) {
                throw std::runtime_error("failed to initialize the GGML CPU backend");
            }
            ggml_backend_cpu_set_n_threads(backend, config.cpu_threads);
            return backend;
        }
        case BackendKind::Vulkan:
#if NANOVLLM_NATIVE_HAS_VULKAN
        {
            const int count = ggml_backend_vk_get_device_count();
            if (count <= 0) {
                throw std::runtime_error(
                    "native runtime was built with Vulkan, but no Vulkan device is available");
            }
            if (config.device_index >= static_cast<std::size_t>(count)) {
                std::ostringstream message;
                message << "Vulkan device index " << config.device_index
                        << " is out of range; " << count << " device(s) are available";
                throw std::runtime_error(message.str());
            }
            ggml_backend_t backend = ggml_backend_vk_init(config.device_index);
            if (backend == nullptr) {
                std::ostringstream message;
                message << "failed to initialize GGML Vulkan device " << config.device_index;
                throw std::runtime_error(message.str());
            }
            return backend;
        }
#else
            throw std::runtime_error(
                "Vulkan backend requested, but NANOVLLM_NATIVE_HAS_VULKAN is disabled");
#endif
    }
    throw std::runtime_error("unknown backend kind");
}

}  // namespace

const char * backend_kind_name(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::Cpu:
            return "cpu";
        case BackendKind::Vulkan:
            return "vulkan";
    }
    return "unknown";
}

BackendKind parse_backend_kind(std::string_view kind) {
    if (kind == "cpu") {
        return BackendKind::Cpu;
    }
    if (kind == "vulkan") {
        return BackendKind::Vulkan;
    }
    throw std::runtime_error("backend kind must be 'cpu' or 'vulkan'");
}

bool vulkan_backend_compiled() noexcept {
    return NANOVLLM_NATIVE_HAS_VULKAN != 0;
}

BackendConfig BackendConfig::cpu(int threads) {
    if (threads <= 0) {
        throw std::runtime_error("CPU thread count must be greater than zero");
    }
    BackendConfig config;
    config.kind = BackendKind::Cpu;
    config.cpu_threads = threads;
    config.device_index = 0;
    return config;
}

BackendConfig BackendConfig::vulkan(std::size_t device_index) {
    BackendConfig config;
    config.kind = BackendKind::Vulkan;
    config.device_index = device_index;
    return config;
}

std::vector<BackendDeviceInfo> available_backend_devices() {
    std::vector<BackendDeviceInfo> devices;
    {
        Backend cpu(BackendConfig::cpu(1));
        devices.push_back(cpu.device_info());
    }

#if NANOVLLM_NATIVE_HAS_VULKAN
    const int count = ggml_backend_vk_get_device_count();
    for (int index = 0; index < count; ++index) {
        std::array<char, 256> description{};
        ggml_backend_vk_get_device_description(index, description.data(), description.size());

        Backend backend(BackendConfig::vulkan(static_cast<std::size_t>(index)));
        BackendDeviceInfo info = backend.device_info();
        if (description[0] != '\0') {
            info.description = description.data();
        }
        ggml_backend_vk_get_device_memory(index, &info.memory_free, &info.memory_total);
        devices.push_back(std::move(info));
    }
#endif
    return devices;
}

Backend::Backend(const BackendConfig & config)
    : handle_(create_backend_handle(config)), config_(config) {
    try {
        if (config_.kind == BackendKind::Cpu) {
            ggml_threadpool_params params =
                ggml_threadpool_params_default(config_.cpu_threads);
            threadpool_ = ggml_threadpool_new(&params);
            if (threadpool_ == nullptr) {
                throw std::runtime_error("failed to create the persistent GGML CPU threadpool");
            }
            ggml_backend_cpu_set_threadpool(handle_, threadpool_);
        }

        std::string fallback_description;
#if NANOVLLM_NATIVE_HAS_VULKAN
        if (config_.kind == BackendKind::Vulkan) {
            std::array<char, 256> description{};
            ggml_backend_vk_get_device_description(
                static_cast<int>(config_.device_index), description.data(), description.size());
            fallback_description = description.data();
        }
#endif
        device_info_ = describe_backend(
            handle_, config_.kind, config_.device_index, fallback_description);
        name_ = safe_string(ggml_backend_name(handle_), backend_kind_name(config_.kind));
    } catch (...) {
        release();
        throw;
    }
}

Backend::~Backend() {
    release();
}

Backend::Backend(Backend && other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      threadpool_(std::exchange(other.threadpool_, nullptr)),
      config_(other.config_),
      device_info_(std::move(other.device_info_)),
      name_(std::move(other.name_)) {}

Backend & Backend::operator=(Backend && other) noexcept {
    if (this != &other) {
        release();
        handle_ = std::exchange(other.handle_, nullptr);
        threadpool_ = std::exchange(other.threadpool_, nullptr);
        config_ = other.config_;
        device_info_ = std::move(other.device_info_);
        name_ = std::move(other.name_);
    }
    return *this;
}

void Backend::release() noexcept {
    if (handle_ != nullptr) {
        if (threadpool_ != nullptr) {
            // The threadpool is externally owned by Backend.  Stop all work and
            // remove the CPU backend's borrowed reference before either object
            // is destroyed.
            ggml_backend_synchronize(handle_);
            ggml_backend_cpu_set_threadpool(handle_, nullptr);
        }
        ggml_backend_free(handle_);
        handle_ = nullptr;
    }
    if (threadpool_ != nullptr) {
        ggml_threadpool_free(threadpool_);
        threadpool_ = nullptr;
    }
}

int Backend::threadpool_threads() const noexcept {
    return threadpool_ == nullptr ? 0 : config_.cpu_threads;
}

ggml_backend_buffer_type_t Backend::default_buffer_type() const {
    require_backend(handle_, "get default buffer type");
    ggml_backend_buffer_type_t buffer_type = ggml_backend_get_default_buffer_type(handle_);
    if (buffer_type == nullptr) {
        throw std::runtime_error(name_ + " backend returned a null default buffer type");
    }
    return buffer_type;
}

bool Backend::supports_buffer_type(ggml_backend_buffer_type_t buffer_type) const noexcept {
    return backend_supports_buffer_type(handle_, buffer_type);
}

void Backend::synchronize() const {
    require_backend(handle_, "synchronize backend");
    ggml_backend_synchronize(handle_);
}

BackendList BackendList::cpu_only(int cpu_threads) {
    BackendList result;
    result.backends_.emplace_back(BackendConfig::cpu(cpu_threads));
    return result;
}

BackendList BackendList::vulkan_with_cpu(int cpu_threads, std::size_t vulkan_device_index) {
    BackendList result;
    result.backends_.reserve(2);
    // Scheduler priority is positional: Vulkan must precede the CPU fallback.
    result.backends_.emplace_back(BackendConfig::vulkan(vulkan_device_index));
    result.backends_.emplace_back(BackendConfig::cpu(cpu_threads));
    return result;
}

Backend & BackendList::at(std::size_t index) {
    if (index >= backends_.size()) {
        throw std::runtime_error("backend index is out of range");
    }
    return backends_[index];
}

const Backend & BackendList::at(std::size_t index) const {
    if (index >= backends_.size()) {
        throw std::runtime_error("backend index is out of range");
    }
    return backends_[index];
}

std::vector<ggml_backend_t> BackendList::raw_handles() const {
    std::vector<ggml_backend_t> result;
    result.reserve(backends_.size());
    for (const Backend & backend : backends_) {
        if (backend.get() == nullptr) {
            throw std::runtime_error("backend list contains a moved-from backend");
        }
        result.push_back(backend.get());
    }
    return result;
}

std::vector<ggml_backend_buffer_type_t> BackendList::default_buffer_types() const {
    std::vector<ggml_backend_buffer_type_t> result;
    result.reserve(backends_.size());
    for (const Backend & backend : backends_) {
        result.push_back(backend.default_buffer_type());
    }
    return result;
}

std::string buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    if (buffer_type == nullptr) {
        return "<null>";
    }
    return safe_string(ggml_backend_buft_name(buffer_type), "<unnamed>");
}

bool backend_supports_buffer_type(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buffer_type) noexcept {
    return backend != nullptr && buffer_type != nullptr &&
           ggml_backend_supports_buft(backend, buffer_type);
}

bool buffer_is_compatible_with_backend(
    ggml_backend_t backend,
    ggml_backend_buffer_t buffer) noexcept {
    return buffer != nullptr &&
           backend_supports_buffer_type(backend, ggml_backend_buffer_get_type(buffer));
}

void assert_buffer_type_compatible(
    ggml_backend_t backend,
    ggml_backend_buffer_type_t buffer_type,
    std::string_view label) {
    require_backend(backend, labelled_message(label, "check buffer compatibility"));
    if (buffer_type == nullptr) {
        throw std::runtime_error(labelled_message(label, "buffer type is null"));
    }
    if (!backend_supports_buffer_type(backend, buffer_type)) {
        std::ostringstream message;
        message << "buffer type '" << buffer_type_name(buffer_type)
                << "' is not supported by backend '"
                << safe_string(ggml_backend_name(backend), "<unnamed>") << "'";
        throw std::runtime_error(labelled_message(label, message.str()));
    }
}

void assert_buffer_on_backend(
    ggml_backend_buffer_t buffer,
    ggml_backend_t backend,
    std::string_view label) {
    if (buffer == nullptr) {
        throw std::runtime_error(labelled_message(label, "backend buffer is null"));
    }
    assert_buffer_type_compatible(backend, ggml_backend_buffer_get_type(buffer), label);
}

void assert_tensor_on_backend(
    const ggml_tensor * tensor,
    ggml_backend_t backend,
    std::string_view label) {
    if (tensor == nullptr) {
        throw std::runtime_error(labelled_message(label, "tensor is null"));
    }
    std::string tensor_label(label);
    if (tensor_label.empty() && tensor->name[0] != '\0') {
        tensor_label = tensor->name;
    }
    if (tensor->buffer == nullptr) {
        throw std::runtime_error(labelled_message(tensor_label, "tensor has no backend buffer"));
    }
    assert_buffer_on_backend(tensor->buffer, backend, tensor_label);
}

}  // namespace nanovllm::native
