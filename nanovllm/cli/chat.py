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
    return prompt


def clean_response(text: str, template: str) -> str:
    if template == "qwen":
        for stop in QWEN_STOPS:
            if stop in text:
                text = text.split(stop, 1)[0]
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive nano-vLLM chat CLI for CUDA or llama.cpp CPU/Vulkan/PD backends.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=os.environ.get("NANOVLLM_GGUF_MODEL") or os.environ.get("NANOVLLM_MODEL"),
        help="HF model path for CUDA, or GGUF model path for llama.cpp backends.",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("NANOVLLM_BACKEND", "llamacpp_cpu"),
        choices=("cuda", "llamacpp_cpu", "llamacpp_vulkan", "llamacpp_pd"),
        help="Execution backend.",
    )
    parser.add_argument(
        "--gguf-model",
        default=os.environ.get("NANOVLLM_GGUF_MODEL"),
        help="GGUF model path. Defaults to the positional model path for llama.cpp backends.",
    )
    parser.add_argument(
        "--library-path",
        default=os.environ.get("NANOVLLM_LLAMA_BACKEND_LIB"),
        help="Path to libnanollama_backend.so.",
    )
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("NANOVLLM_MAX_MODEL_LEN", "2048")))
    parser.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("NANOVLLM_MAX_NUM_SEQS", "8")))
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=int(os.environ.get("NANOVLLM_MAX_NUM_BATCHED_TOKENS", "2048")),
    )
    parser.add_argument(
        "--ubatch-size",
        type=int,
        default=int(os.environ.get("NANOVLLM_UBATCH_SIZE", "0")),
        help="llama.cpp physical ubatch size. 0 uses backend defaults; Vulkan/PD defaults to min(batch, 512).",
    )
    parser.add_argument("--threads", type=int, default=int(os.environ.get("NANOVLLM_THREADS", "8")))
    parser.add_argument(
        "--threads-batch",
        type=int,
        default=int(os.environ.get("NANOVLLM_THREADS_BATCH", "8")),
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=int(os.environ.get("NANOVLLM_GPU_LAYERS", "-1")),
        help="Number of layers to offload for llama.cpp Vulkan/PD prefill. -1 means all supported layers.",
    )
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("NANOVLLM_TEMPERATURE", "0.6")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("NANOVLLM_MAX_TOKENS", "256")))
    parser.add_argument(
        "--system",
        default=os.environ.get("NANOVLLM_SYSTEM_PROMPT", "You are a helpful assistant."),
        help="System prompt used by the chat template.",
    )
    parser.add_argument(
        "--template",
        default=os.environ.get("NANOVLLM_CHAT_TEMPLATE", "qwen"),
        choices=("qwen", "plain"),
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
    model = args.model
    gguf_model = args.gguf_model or model
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
            device_config={
                "library_path": args.library_path,
                "n_ubatch": args.ubatch_size or None,
                "n_threads": args.threads,
                "n_threads_batch": args.threads_batch,
                "n_gpu_layers": args.gpu_layers,
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


def print_help() -> None:
    print("Commands: /exit, /quit, /reset, /history, /system <prompt>, /help")


def main() -> int:
    args = parse_args()
    llm = build_llm(args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    messages: list[dict[str, str]] = []
    system_prompt = args.system

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
                print("system prompt updated and history cleared")
                continue

            messages.append({"role": "user", "content": user_input})
            prompt = format_chat_prompt(messages, system_prompt, args.template)
            outputs = llm.generate([prompt], sampling_params, use_tqdm=args.tqdm)
            response = clean_response(outputs[0]["text"], args.template)
            messages.append({"role": "assistant", "content": response})
            print(f"\nassistant> {response}")
    finally:
        if hasattr(llm, "exit"):
            llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
