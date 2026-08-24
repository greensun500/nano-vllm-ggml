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


def format_turn_metrics(metrics: dict[str, int | float]) -> str:
    def stage(label: str, token_key: str, second_key: str) -> str:
        tokens = int(metrics.get(token_key, 0))
        seconds = float(metrics.get(second_key, 0.0))
        if tokens <= 0 or seconds <= 0.0:
            return f"{label}: n/a"
        return f"{label}: {tokens / seconds:.2f} tok/s ({tokens} tok, {seconds * 1000.0:.1f} ms)"

    generated = int(metrics.get("generated_tokens", 0))
    prefill_generated = int(metrics.get("prefill_generated_tokens", 0))
    decode_generated = int(metrics.get("decode_tokens", 0))
    generated_summary = f"generated: {generated} tok"
    if generated == prefill_generated + decode_generated:
        generated_summary += f" (prefill {prefill_generated} + decode {decode_generated})"
    return "[timing] " + " | ".join((
        stage("prefill", "prefill_tokens", "prefill_seconds"),
        stage("decode", "decode_tokens", "decode_seconds"),
        generated_summary,
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive nanovllm-ggml chat CLI for native CPU, Vulkan, or CUDA backends.",
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
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("NANOVLLM_MAX_MODEL_LEN", "2048")))
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
    parser.add_argument("--threads", type=int, default=int(os.environ.get("NANOVLLM_THREADS", "8")))
    parser.add_argument("--device-index", type=int, default=int(os.environ.get("NANOVLLM_DEVICE_INDEX", "0")))
    mtp_group = parser.add_mutually_exclusive_group()
    mtp_default = (
        os.environ.get("NANOVLLM_ENABLE_MTP", "").lower() == "1"
        if "NANOVLLM_ENABLE_MTP" in os.environ
        else False
    )
    mtp_group.add_argument(
        "--enable-mtp",
        action="store_true",
        dest="enable_mtp",
        default=mtp_default,
        help="Enable built-in Qwen3.5 MTP greedy speculative decoding.",
    )
    mtp_group.add_argument(
        "--disable-mtp",
        action="store_false",
        dest="enable_mtp",
        help="Disable MTP even when the native device auto profile recommends it.",
    )
    parser.add_argument(
        "--mtp-max-draft-tokens",
        type=int,
        default=int(os.environ.get("NANOVLLM_MTP_MAX_DRAFT_TOKENS", "1")),
    )
    parser.add_argument(
        "--mali-experimental-graph-reuse",
        action="store_true",
        default=os.environ.get("NANOVLLM_MALI_EXPERIMENTAL_GRAPH_REUSE", "0") == "1",
        help="Experimental: enable persistent graph reuse on Mali Vulkan.",
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
        help="Show nanovllm-ggml generation progress for each turn.",
    )
    return parser.parse_args()


def build_llm(args: argparse.Namespace) -> LLM:
    model = args.model or args.gguf_model
    gguf_model = args.gguf_model or model
    if not gguf_model:
        raise SystemExit("Please pass a GGUF model path or set NANOVLLM_GGUF_MODEL.")
    return LLM(
        gguf_model,
        backend=args.backend,
        gguf_model=gguf_model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enable_mtp=args.enable_mtp,
        mtp_max_draft_tokens=args.mtp_max_draft_tokens,
        mali_experimental_graph_reuse=args.mali_experimental_graph_reuse,
        device_config={"n_threads": args.threads, "device_index": args.device_index},
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive.")
    if args.temperature != 0:
        raise SystemExit("CPU/Vulkan GGUF backends currently require --temperature 0.")


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

    print(f"nanovllm-ggml chat ready. backend={args.backend}, template={args.template}")
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
            metrics = output.get("metrics") if args.backend.startswith("native") else None
            if isinstance(metrics, dict):
                print(format_turn_metrics(metrics))
    finally:
        if session is not None and session.is_open:
            session.close()
        if hasattr(llm, "exit"):
            llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
