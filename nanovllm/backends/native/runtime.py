"""Thin Python API for nano-vLLM's in-tree GGML runtime.

The compiled module is deliberately imported as part of the nano-vLLM package.
It statically embeds GGML CPU/Vulkan and never loads ``llama_context`` or an
external ``libnanollama_backend.so``.
"""


def _extension():
    try:
        from nanovllm import _C
    except ImportError as exc:
        raise RuntimeError(
            "nano-vLLM native runtime is not built. Run "
            "scripts/build_native_runtime.sh first."
        ) from exc
    return _C


def build_info() -> dict:
    return _extension().build_info()


def available_backends() -> list[dict]:
    return _extension().available_backends()


def gguf_info(path: str) -> dict:
    return _extension().gguf_info(path=path)


def matmul_smoke(backend: str = "cpu", threads: int = 1) -> dict:
    return _extension().matmul_smoke(backend=backend, threads=threads)


def q4_0_matmul_smoke(backend: str = "cpu", threads: int = 1) -> dict:
    return _extension().q4_0_matmul_smoke(backend=backend, threads=threads)


def qwen35_weight_load_smoke(path: str, backend: str = "cpu", threads: int = 1) -> dict:
    return _extension().qwen35_weight_load_smoke(
        path=path,
        backend=backend,
        threads=threads,
    )


def qwen35_model_info(path: str) -> dict:
    return _extension().qwen35_model_info(path=path)
