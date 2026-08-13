import argparse
import os
import sys

from nanovllm import LLM, SamplingParams


QWEN_STOPS = ("<|im_end|>", "<|endoftext|>")


def format_chat_prompt(messages: list[dict[str, str]], system_prompt: str, template: str) -> str:
    if template == "plain":
        lines = []
        if system_prompt:
            lines.append(f"System: {system_prompt}")
        for message in messages:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f"{role}: {message['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    prompt = ""
    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for message in messages:
        prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    if template == "qwen35":
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def format_next_turn(user_text: str, template: str) -> str:
    if template == "plain":
        return f"\nUser: {user_text}\nAssistant:"
    prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    if template == "qwen35":
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def clean_response(text: str, template: str) -> str:
    if template in ("qwen", "qwen35"):
        for stop in QWEN_STOPS:
            if stop in text:
                text = text.split(stop, 1)[0]
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive nano-vLLM chat CLI for native CPU, Vulkan, or CUDA backends.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="GGUF model path for the native backend.",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("NANOVLLM_BACKEND", "native_cpu"),
        choices=("native_cpu", "native_vulkan", "native_cuda"),
        help="Execution backend.",
    )
    parser.add_argument(
        "--gguf-model",
        default=os.environ.get("NANOVLLM_GGUF_MODEL"),
        help="GGUF model path. Defaults to the positional model path.",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("NANOVLLM_TOKENIZER"),
        help="Local Hugging Face tokenizer directory required by native backends.",
    )
    parser.add_argument(
        "--library-path",
        default=os.environ.get("NANOVLLM_LLAMA_BACKEND_LIB"),
        help="Path to libnanollama_backend.so (legacy staged llama.cpp backend only).",
    )
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("NANOVLLM_MAX_MODEL_LEN", "2048")))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("NANOVLLM_MAX_NUM_SEQS", "1")))
    parser.add_argument(
        "--num-kvcache-blocks",
        type=int,
        default=int(os.environ.get("NANOVLLM_NUM_KVCACHE_BLOCKS", "-1")),
        help="Total physical KV blocks. -1 derives the minimum from max-model-len.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=int(os.environ.get("NANOVLLM_MAX_NUM_BATCHED_TOKENS", "2048")),
    )
    parser.add_argument(
        "--ubatch-size",
        type=int,
        default=int(os.environ.get("NANOVLLM_UBATCH_SIZE", "0")),
        help="llama.cpp physical ubatch size. 0 uses backend defaults; Vulkan defaults to min(batch, 512).",
    )
    parser.add_argument("--threads", type=int, default=int(os.environ.get("NANOVLLM_THREADS", "8")))
    parser.add_argument("--device-index", type=int, default=int(os.environ.get("NANOVLLM_DEVICE_INDEX", "0")))
    parser.add_argument(
        "--threads-batch",
        type=int,
        default=int(os.environ.get("NANOVLLM_THREADS_BATCH", "8")),
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=int(os.environ.get("NANOVLLM_GPU_LAYERS", "-1")),
        help="Number of layers to offload with Vulkan. -1 means all supported layers.",
    )
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=os.environ.get("NANOVLLM_ENABLE_MTP", "0") == "1",
        help="Enable built-in Qwen3.5 MTP greedy speculative decoding.",
    )
    parser.add_argument(
        "--mtp-max-draft-tokens",
        type=int,
        default=int(os.environ.get("NANOVLLM_MTP_MAX_DRAFT_TOKENS", "3")),
    )
    parser.add_argument(
        "--no-graph-reuse",
        action="store_true",
        default=os.environ.get("NANOVLLM_NO_GRAPH_REUSE", "0") == "1",
        help="Disable native persistent graph bucket reuse.",
    )
    parser.add_argument(
        "--native-vulkan-graph-reuse",
        action="store_true",
        default=os.environ.get("NANOVLLM_NATIVE_VULKAN_GRAPH_REUSE", "0") == "1",
        help="Experimental: reuse stable single-sequence Vulkan decode/MTP graph buckets.",
    )
    parser.add_argument(
        "--native-attention-impl",
        choices=("math", "auto", "flash"),
        default=os.environ.get("NANOVLLM_NATIVE_ATTENTION_IMPL", "math"),
        help="Native attention implementation. 'auto' probes FlashAttention for non-MTP runs and keeps math for MTP.",
    )
    parser.add_argument(
        "--native-batched-recurrent-snapshots",
        action="store_true",
        default=os.environ.get("NANOVLLM_NATIVE_BATCHED_RECURRENT_SNAPSHOTS", "0") == "1",
        help="Write contiguous recurrent delta snapshots with one GGML copy.",
    )
    parser.add_argument(
        "--native-mtp-prefill-fusion",
        action="store_true",
        default=os.environ.get("NANOVLLM_NATIVE_MTP_PREFILL_FUSION", "0") == "1",
        help="Fuse native MTP prefill hidden-to-KV maintenance into the target graph.",
    )
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("NANOVLLM_TEMPERATURE", "0.0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("NANOVLLM_MAX_TOKENS", "256")))
    parser.add_argument(
        "--system",
        default=os.environ.get("NANOVLLM_SYSTEM_PROMPT", "You are a helpful assistant."),
        help="System prompt used by the chat template.",
    )
    parser.add_argument(
        "--template",
        default=os.environ.get("NANOVLLM_CHAT_TEMPLATE", "qwen35"),
        choices=("qwen35", "qwen", "plain"),
        help="Prompt template used to convert chat history into one generation prompt.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Keep generating until max_tokens even if EOS is sampled.",
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Show nano-vLLM generation progress for each turn.",
    )
    return parser.parse_args()


def build_llm(args: argparse.Namespace) -> LLM:
    model = args.model or (
        os.environ.get("NANOVLLM_MODEL")
        if args.backend == "cuda"
        else args.gguf_model
    )
    gguf_model = args.gguf_model or model
    if args.backend.startswith("native"):
        if not gguf_model:
            raise SystemExit("Please pass a GGUF model path or set NANOVLLM_GGUF_MODEL.")
        if not args.tokenizer:
            raise SystemExit("Native backends require --tokenizer or NANOVLLM_TOKENIZER pointing to the HF tokenizer directory.")
        return LLM(
            gguf_model,
            backend=args.backend,
            model_format="gguf",
            gguf_model=gguf_model,
            tokenizer=args.tokenizer,
            tokenizer_backend="hf",
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            num_kvcache_blocks=args.num_kvcache_blocks,
            enable_mtp=args.enable_mtp,
            mtp_max_draft_tokens=args.mtp_max_draft_tokens,
            enable_graph_reuse=not args.no_graph_reuse,
            native_vulkan_graph_reuse=args.native_vulkan_graph_reuse,
            native_attention_impl=args.native_attention_impl,
            native_batched_recurrent_snapshots=args.native_batched_recurrent_snapshots,
            native_mtp_prefill_fusion=args.native_mtp_prefill_fusion,
            device_config={
                "n_threads": args.threads,
                "device_index": args.device_index,
            },
        )
    if args.backend.startswith("llamacpp"):
        if not gguf_model:
            raise SystemExit("Please pass a GGUF model path or set NANOVLLM_GGUF_MODEL.")
        return LLM(
            gguf_model,
            backend=args.backend,
            model_format="gguf",
            gguf_model=gguf_model,
            tokenizer_backend="llamacpp",
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            num_kvcache_blocks=args.num_kvcache_blocks,
            enable_mtp=args.enable_mtp,
            mtp_max_draft_tokens=args.mtp_max_draft_tokens,
            device_config={
                "library_path": args.library_path,
                "n_ubatch": args.ubatch_size or None,
                "n_threads": args.threads,
                "n_threads_batch": args.threads_batch,
                "n_gpu_layers": 0 if args.backend == "llamacpp_cpu" else args.gpu_layers,
            },
        )

    if not model:
        raise SystemExit("Please pass an HF model path or set NANOVLLM_MODEL.")
    return LLM(
        model,
        backend="cuda",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive.")
    if args.backend != "cuda" and args.temperature != 0:
        raise SystemExit("CPU/Vulkan GGUF backends currently require --temperature 0.")
    if args.backend == "cuda" and args.enable_mtp:
        raise SystemExit("--enable-mtp is supported only by CPU/Vulkan GGUF backends.")


def print_help() -> None:
    print("Commands: /exit, /quit, /reset, /history, /system <prompt>, /help")


def main() -> int:
    args = parse_args()
    validate_args(args)
    llm = build_llm(args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    messages: list[dict[str, str]] = []
    system_prompt = args.system
    session = None

    print(f"nano-vLLM chat ready. backend={args.backend}, template={args.template}")
    print_help()
    try:
        while True:
            try:
                user_input = input("\nuser> ").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue
            if user_input in ("/exit", "/quit"):
                break
            if user_input == "/help":
                print_help()
                continue
            if user_input == "/reset":
                messages.clear()
                if session is not None:
                    session.close()
                    session = None
                print("history cleared")
                continue
            if user_input == "/history":
                if not messages:
                    print("(empty)")
                for message in messages:
                    print(f"{message['role']}> {message['content']}")
                continue
            if user_input.startswith("/system "):
                system_prompt = user_input[len("/system ") :].strip()
                messages.clear()
                if session is not None:
                    session.close()
                    session = None
                print("system prompt updated and history cleared")
                continue

            messages.append({"role": "user", "content": user_input})
            if args.backend.startswith("native"):
                if session is None:
                    prompt = format_chat_prompt(messages, system_prompt, args.template)
                    session, output = llm.start_session(prompt, sampling_params, use_tqdm=args.tqdm)
                else:
                    output = session.generate(
                        format_next_turn(user_input, args.template),
                        sampling_params,
                        use_tqdm=args.tqdm,
                    )
                response = clean_response(output["text"], args.template)
            else:
                prompt = format_chat_prompt(messages, system_prompt, args.template)
                outputs = llm.generate([prompt], sampling_params, use_tqdm=args.tqdm)
                response = clean_response(outputs[0]["text"], args.template)
            messages.append({"role": "assistant", "content": response})
            print(f"\nassistant> {response}")
    finally:
        if session is not None and session.is_open:
            session.close()
        if hasattr(llm, "exit"):
            llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
