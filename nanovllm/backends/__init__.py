from nanovllm.config import Config


def create_backend(config: Config, rank: int = 0, event=None):
    if config.backend == "cuda":
        from nanovllm.engine.model_runner import ModelRunner

        return ModelRunner(config, rank, event)
    if config.backend in ("llamacpp_cpu", "llamacpp_vulkan"):
        from nanovllm.backends.llamacpp.runner import LlamaCppRunner

        return LlamaCppRunner(config)
    raise ValueError(f"unsupported backend: {config.backend}")
