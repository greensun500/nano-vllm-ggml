import os

from nanovllm import LLM, SamplingParams


def main():
    gguf_model = os.environ.get("NANOVLLM_GGUF_MODEL", "/home/cix/Qwen2.5-3B-Instruct-Q4_0.gguf")
    backend = os.environ.get("NANOVLLM_BACKEND", "llamacpp_cpu")
    library_path = os.environ.get("NANOVLLM_LLAMA_BACKEND_LIB")

    llm = LLM(
        gguf_model,
        backend=backend,
        model_format="gguf",
        gguf_model=gguf_model,
        tokenizer_backend="llamacpp",
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        device_config={
            "library_path": library_path,
            "n_ubatch": int(os.environ.get("NANOVLLM_UBATCH_SIZE", "0")) or None,
            "n_threads": int(os.environ.get("NANOVLLM_THREADS", "8")),
            "n_threads_batch": int(os.environ.get("NANOVLLM_THREADS_BATCH", "8")),
        },
    )
    outputs = llm.generate(
        ["introduce yourself"],
        SamplingParams(temperature=0.6, max_tokens=64),
        use_tqdm=True,
    )
    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
