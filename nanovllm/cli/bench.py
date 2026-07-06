import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from time import perf_counter

from nanovllm import LLM, SamplingParams
from nanovllm.backends.base import build_execution_plan


@dataclass(slots=True)
class BenchResult:
    backend: str
    model: str
    batch_size: int
    prompt_len: int
    gen_len: int
    use_prefix_cache: bool
    repeat: int
    load_s: float
    load_rss_mib: float
    total_s: float
    prefill_s: float
    decode_s: float
    prefill_tokens: int
    decode_tokens: int
    generated_tokens: int
    prefill_tok_s: float
    decode_tok_s: float
    generated_tok_s: float
    shared_kv_mib: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="nano-vLLM benchmark CLI for CUDA and llama.cpp CPU/Vulkan/PD backends.",
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
    parser.add_argument("--gguf-model", default=os.environ.get("NANOVLLM_GGUF_MODEL"))
    parser.add_argument("--library-path", default=os.environ.get("NANOVLLM_LLAMA_BACKEND_LIB"))
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
    parser.add_argument("--threads-batch", type=int, default=int(os.environ.get("NANOVLLM_THREADS_BATCH", "8")))
    parser.add_argument("--gpu-layers", type=int, default=int(os.environ.get("NANOVLLM_GPU_LAYERS", "-1")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("NANOVLLM_BENCH_BATCH", "1")))
    parser.add_argument("--prompt-len", type=int, default=int(os.environ.get("NANOVLLM_BENCH_PROMPT_LEN", "128")))
    parser.add_argument("--gen-len", type=int, default=int(os.environ.get("NANOVLLM_BENCH_GEN_LEN", "32")))
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("NANOVLLM_BENCH_REPEAT", "3")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("NANOVLLM_BENCH_WARMUP", "1")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("NANOVLLM_TEMPERATURE", "0.6")))
    parser.add_argument(
        "--use-prefix-cache",
        action="store_true",
        help="Reuse identical prompts across runs to benchmark prefix-cache behavior.",
    )
    parser.add_argument(
        "--prompt",
        default=os.environ.get(
            "NANOVLLM_BENCH_PROMPT",
            "Benchmark prompt for nano-vLLM performance measurement. ",
        ),
        help="Seed text repeated/truncated to build synthetic prompt token IDs.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
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


def current_rss_mib() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            fields = f.read().split()
        resident_pages = int(fields[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024.0 / 1024.0
    except (OSError, IndexError, ValueError):
        return 0.0


def encode_prompt(llm: LLM, text: str) -> list[int]:
    if llm.config.tokenizer_backend == "llamacpp":
        return llm.model_runner.call("tokenize", text)
    return llm.tokenizer.encode(text)


def make_prompt_token_ids(llm: LLM, seed_text: str, prompt_len: int) -> list[int]:
    if prompt_len <= 0:
        raise SystemExit("--prompt-len must be positive.")
    seed_tokens = encode_prompt(llm, seed_text)
    if not seed_tokens:
        raise SystemExit("The benchmark prompt produced no tokens.")
    repeats = (prompt_len + len(seed_tokens) - 1) // len(seed_tokens)
    return (seed_tokens * repeats)[:prompt_len]


def get_vocab_size(llm: LLM) -> int:
    if llm.config.tokenizer_backend == "llamacpp":
        return int(llm.model_runner.vocab_size)
    return int(llm.config.hf_config.vocab_size)


def make_batch_prompts(
    base_prompt: list[int],
    batch_size: int,
    vocab_size: int,
    run_index: int,
    use_prefix_cache: bool,
) -> list[list[int]]:
    prompts = [list(base_prompt) for _ in range(batch_size)]
    if use_prefix_cache:
        return prompts
    for i, prompt in enumerate(prompts):
        prompt[0] = (prompt[0] + run_index * batch_size + i + 1) % vocab_size
    return prompts


def run_generation_once(
    llm: LLM,
    prompts: list[list[int]],
    sampling_params: list[SamplingParams],
) -> dict[str, float | int]:
    for prompt, sp in zip(prompts, sampling_params):
        llm.add_request(prompt, sp)

    prefill_s = 0.0
    decode_s = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    generated_tokens = 0
    total_start = perf_counter()

    while not llm.is_finished():
        t0 = perf_counter()
        seqs, is_prefill = llm.scheduler.schedule()
        llm.flush_backend_releases()
        plan = build_execution_plan(seqs, is_prefill, llm.config.kvcache_block_size)
        scheduled_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)
        completion_counts = [seq.num_completion_tokens for seq in seqs]
        token_ids = llm.model_runner.call("run", plan)
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        llm.flush_backend_releases()
        elapsed = perf_counter() - t0

        generated_tokens += sum(seq.num_completion_tokens - before for seq, before in zip(seqs, completion_counts))
        if is_prefill:
            prefill_s += elapsed
            prefill_tokens += scheduled_tokens
        else:
            decode_s += elapsed
            decode_tokens += scheduled_tokens

    return {
        "total_s": perf_counter() - total_start,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "generated_tokens": generated_tokens,
    }


def benchmark(args: argparse.Namespace) -> BenchResult:
    if args.batch_size <= 0 or args.gen_len <= 0 or args.repeat <= 0 or args.warmup < 0:
        raise SystemExit("--batch-size, --gen-len and --repeat must be positive; --warmup must be non-negative.")
    if args.batch_size > args.max_num_seqs:
        raise SystemExit("--batch-size must be <= --max-num-seqs.")

    load_start = perf_counter()
    llm = build_llm(args)
    load_s = perf_counter() - load_start
    load_rss_mib = current_rss_mib()

    prompt_ids = make_prompt_token_ids(llm, args.prompt, args.prompt_len)
    vocab_size = get_vocab_size(llm)
    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.gen_len)
        for _ in range(args.batch_size)
    ]

    shared_kv_mib = float(getattr(llm.model_runner, "shared_kv_bytes", 0)) / 1024.0 / 1024.0

    try:
        run_index = 0
        for _ in range(args.warmup):
            prompts = make_batch_prompts(prompt_ids, args.batch_size, vocab_size, run_index, args.use_prefix_cache)
            run_generation_once(llm, prompts, sampling_params)
            run_index += 1

        totals = {
            "total_s": 0.0,
            "prefill_s": 0.0,
            "decode_s": 0.0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "generated_tokens": 0,
        }
        for _ in range(args.repeat):
            prompts = make_batch_prompts(prompt_ids, args.batch_size, vocab_size, run_index, args.use_prefix_cache)
            result = run_generation_once(llm, prompts, sampling_params)
            run_index += 1
            for key in totals:
                totals[key] += result[key]
    finally:
        llm.exit()

    prefill_tok_s = totals["prefill_tokens"] / totals["prefill_s"] if totals["prefill_s"] > 0 else 0.0
    decode_tok_s = totals["decode_tokens"] / totals["decode_s"] if totals["decode_s"] > 0 else 0.0
    generated_tok_s = totals["generated_tokens"] / totals["total_s"] if totals["total_s"] > 0 else 0.0
    return BenchResult(
        backend=args.backend,
        model=args.gguf_model or args.model or "",
        batch_size=args.batch_size,
        prompt_len=len(prompt_ids),
        gen_len=args.gen_len,
        use_prefix_cache=args.use_prefix_cache,
        repeat=args.repeat,
        load_s=load_s,
        load_rss_mib=load_rss_mib,
        total_s=totals["total_s"],
        prefill_s=totals["prefill_s"],
        decode_s=totals["decode_s"],
        prefill_tokens=totals["prefill_tokens"],
        decode_tokens=totals["decode_tokens"],
        generated_tokens=totals["generated_tokens"],
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        generated_tok_s=generated_tok_s,
        shared_kv_mib=shared_kv_mib,
    )


def print_result(result: BenchResult) -> None:
    print("nano-vLLM bench")
    print(f"backend:            {result.backend}")
    print(f"model:              {result.model}")
    print(f"batch size:         {result.batch_size}")
    print(f"prompt tokens/seq:  {result.prompt_len}")
    print(f"gen tokens/seq:     {result.gen_len}")
    print(f"prefix cache:       {result.use_prefix_cache}")
    print(f"repeat:             {result.repeat}")
    print(f"load time:          {result.load_s:.3f} s")
    if result.load_rss_mib > 0:
        print(f"load RSS:           {result.load_rss_mib:.2f} MiB")
    if result.shared_kv_mib > 0:
        print(f"shared KV pool:     {result.shared_kv_mib:.2f} MiB")
    print()
    print("metric              tokens        time(s)      tok/s")
    print(f"prefill             {result.prefill_tokens:>8}  {result.prefill_s:>10.3f}  {result.prefill_tok_s:>9.2f}")
    print(f"decode steps        {result.decode_tokens:>8}  {result.decode_s:>10.3f}  {result.decode_tok_s:>9.2f}")
    print(f"generated total     {result.generated_tokens:>8}  {result.total_s:>10.3f}  {result.generated_tok_s:>9.2f}")


def main() -> int:
    args = parse_args()
    result = benchmark(args)
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
