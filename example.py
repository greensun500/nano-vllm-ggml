"""Minimal GGUF-only native inference example."""

from __future__ import annotations

import os

from nanovllm import LLM, SamplingParams


def main() -> None:
    model = os.environ["NANOVLLM_QWEN35_GGUF"]
    llm = LLM(
        model,
        backend=os.environ.get("NANOVLLM_BACKEND", "native_cpu"),
        max_model_len=512,
        num_kvcache_blocks=2,
        device_config={"n_threads": int(os.environ.get("NANOVLLM_THREADS", "4"))},
    )
    try:
        result = llm.generate(
            ["请用一句话介绍 nanovllm-ggml。"],
            SamplingParams(temperature=0, max_tokens=32),
            use_tqdm=False,
        )
        print(result[0]["text"])
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
