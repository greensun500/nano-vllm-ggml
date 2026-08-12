import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    # 1. 模型路径
    #
    # 原始 nano-vLLM 示例使用 Hugging Face 目录加载 CUDA/PyTorch 模型。
    # 当前 Qwen3.5 native 路径使用 GGUF 权重执行计算，同时使用匹配的
    # Hugging Face tokenizer 目录做 chat template / encode / decode。
    #
    # 可以通过环境变量覆盖，方便在远程机器使用不同路径：
    #
    #   NANOVLLM_QWEN35_GGUF=/path/to/Qwen3.5-2B-Q4_0.gguf
    #   NANOVLLM_QWEN35_TOKENIZER=/path/to/Qwen3.5-2B
    #
    gguf_model = os.environ.get(
        "NANOVLLM_QWEN35_GGUF",
        "/home/kevin/model_raw/Qwen3.5-2B-Q4_0.gguf",
    )
    tokenizer_path = os.environ.get(
        "NANOVLLM_QWEN35_TOKENIZER",
        "/home/kevin/model_raw/Qwen3.5-2B",
    )

    # 2. 加载 tokenizer
    #
    # native backend 当前要求 tokenizer_backend="hf"，所以 tokenizer 仍在
    # Python/Transformers 侧负责。模型推理本身不走 transformers。
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # 3. 初始化 LLM
    #
    # 这里是相对原始 example.py 最关键的变化：
    #
    # - backend="native_cpu"：
    #   走我们新增的 in-tree C++ Qwen35Runtime，底层使用 GGML CPU backend。
    #
    # - 如果要测 Vulkan，把 backend 改成 "native_vulkan"，并先用
    #   NANOVLLM_NATIVE_VULKAN=ON ./scripts/build_native_runtime.sh 构建。
    #
    # - model_format="gguf" / gguf_model=...：
    #   告诉 Config 这是 GGUF 模型，不是原始 HF 权重目录。
    #
    # - tokenizer=... / tokenizer_backend="hf"：
    #   native runtime 只负责模型计算，chat template 和 decode 仍用 HF tokenizer。
    #
    # - enable_mtp：
    #   是否开启 Qwen3.5 GGUF 内置 MTP speculative decoding。
    #   第一版只支持 greedy，所以 SamplingParams.temperature 需要是 0。
    #
    # - enable_graph_reuse：
    #   v3.5 的 CPU 单序列 fixed-shape graph reuse。Vulkan 当前会 fallback。
    llm = LLM(
        gguf_model,
        backend=os.environ.get("NANOVLLM_BACKEND", "native_cpu"),
        model_format="gguf",
        gguf_model=gguf_model,
        tokenizer=tokenizer_path,
        tokenizer_backend="hf",
        max_model_len=int(os.environ.get("NANOVLLM_MAX_MODEL_LEN", "512")),
        max_num_batched_tokens=int(os.environ.get("NANOVLLM_MAX_NUM_BATCHED_TOKENS", "512")),
        max_num_seqs=int(os.environ.get("NANOVLLM_MAX_NUM_SEQS", "1")),
        num_kvcache_blocks=int(os.environ.get("NANOVLLM_NUM_KVCACHE_BLOCKS", "-1")),
        enable_mtp=os.environ.get("NANOVLLM_ENABLE_MTP", "0") == "1",
        mtp_max_draft_tokens=int(os.environ.get("NANOVLLM_MTP_MAX_DRAFT_TOKENS", "3")),
        enable_graph_reuse=os.environ.get("NANOVLLM_NO_GRAPH_REUSE", "0") != "1",
        device_config={
            "n_threads": int(os.environ.get("NANOVLLM_THREADS", str(os.cpu_count() or 1))),
            "device_index": int(os.environ.get("NANOVLLM_DEVICE_INDEX", "0")),
        },
    )

    # 4. 采样参数
    #
    # temperature=0 表示 greedy。当前 native Qwen3.5/MTP 路径只支持 greedy。
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=int(os.environ.get("NANOVLLM_MAX_TOKENS", "64")),
    )

    # 5. 准备 prompt
    #
    # Qwen3.5 是 chat/instruct 模型，建议通过 tokenizer.apply_chat_template
    # 生成模型真实看到的 prompt。
    raw_prompts = [
        "请用三句话介绍一下你自己。",
        "请列出 100 以内的所有质数。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in raw_prompts
    ]

    # 6. 开始推理
    #
    # generate 内部流程：
    #
    #   LLMEngine
    #     -> Scheduler.schedule()
    #     -> build_execution_plan()
    #     -> NativeRunner.run()
    #     -> nanovllm._C.Qwen35Runtime.run()/run_mtp()
    #     -> Scheduler.postprocess()
    #
    outputs = llm.generate(prompts, sampling_params)

    # 7. 打印结果
    for raw_prompt, prompt, output in zip(raw_prompts, prompts, outputs):
        print("\n")
        print(f"Raw prompt: {raw_prompt}")
        print(f"Formatted prompt: {prompt!r}")
        print(f"Token IDs: {output['token_ids']}")
        print(f"Completion: {output['text']!r}")

    # 8. 可选统计
    #
    # NativeRunner 暴露 MTP 统计和 v3.5 graph reuse 统计。CUDA runner 没有这些
    # 方法，所以这里用 getattr 做保护。
    runner = getattr(llm, "model_runner", None)
    mtp_stats = getattr(runner, "mtp_stats", None)
    if callable(mtp_stats):
        print(f"\nMTP stats: {mtp_stats()}")
    graph_reuse_stats = getattr(runner, "graph_reuse_stats", None)
    if callable(graph_reuse_stats):
        print(f"Graph reuse stats: {graph_reuse_stats()}")


if __name__ == "__main__":
    main()
