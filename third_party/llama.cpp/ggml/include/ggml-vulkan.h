#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// Mali G720 MMQ tuning keeps the upstream pipeline as a correctness baseline
// and builds a second, tile-tuned Q8_1 MMQ pipeline. The selection is
// thread-local so a caller can bracket multi-token work with existing KV (or
// another shape known to be unsafe) without changing device-global state.
// It is a no-op on devices without the tuned pipelines. Set
// GGML_VK_DISABLE_MALI_MMQ_TUNE=1 to force the upstream safe pipeline.
GGML_BACKEND_API void ggml_vk_set_mali_mmq_safe_mode(bool safe);

#ifdef  __cplusplus
}
#endif
