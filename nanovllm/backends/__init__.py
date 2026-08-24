from nanovllm.config import Config


def create_backend(config: Config, rank: int = 0, event=None):
    if config.backend in ("native_cpu", "native_vulkan", "native_cuda"):
        from nanovllm.backends.native.runner import NativeRunner

        return NativeRunner(config)
    raise ValueError(f"unsupported backend: {config.backend}")
