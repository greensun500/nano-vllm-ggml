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
    warmup: int
    threads: int
    device_index: int
    num_kvcache_blocks: int
    enable_mtp: bool
    mtp_max_draft_tokens: int
    mtp_drafted_tokens: int
    mtp_accepted_tokens: int
    mtp_verification_steps: int
    mtp_acceptance_rate: float
    mtp_draft_s: float
    mtp_verification_s: float
    mtp_kv_update_s: float
    mtp_draft_graph_setup_s: float
    mtp_verification_graph_setup_s: float
    mtp_kv_update_graph_setup_s: float
    mtp_draft_tok_s: float
    mtp_verification_tok_s: float
    mtp_kv_update_tok_s: float
    enable_graph_reuse: bool
    native_vulkan_graph_reuse: bool
    native_attention_impl: str
    native_batched_recurrent_snapshots: bool
    native_mtp_prefill_fusion: bool
    native_mtp_verification_kv_fusion: bool
    graph_cache_hits: int
    graph_cache_misses: int
    graph_cache_evictions: int
    graph_cache_active_entries: int
    ggml_commit: str
    vulkan_compiled: bool
    cuda_compiled: bool
    cuda_graphs_compiled: bool
    load_s: float
    load_rss_mib: float
    total_s: float
    prefill_s: float
    decode_s: float
    prefill_tokens: int
    decode_tokens: int
    processed_tokens: int
    generated_tokens: int
    prefill_tok_s: float
    decode_tok_s: float
    processed_tok_s: float
    generated_tok_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="nano-vLLM benchmark CLI for native CPU, Vulkan, and CUDA backends.",
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
    parser.add_argument("--gguf-model", default=os.environ.get("NANOVLLM_GGUF_MODEL"))
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("NANOVLLM_TOKENIZER"),
        help="Local Hugging Face tokenizer directory required by native backends.",
    )
    parser.add_argument(
        "--library-path",
        default=os.environ.get("NANOVLLM_LLAMA_BACKEND_LIB"),
        help="External backend library path for legacy llamacpp_* backends only.",
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
    parser.add_argument("--threads-batch", type=int, default=int(os.environ.get("NANOVLLM_THREADS_BATCH", "8")))
    parser.add_argument("--gpu-layers", type=int, default=int(os.environ.get("NANOVLLM_GPU_LAYERS", "-1")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("NANOVLLM_BENCH_BATCH", "1")))
    parser.add_argument("--prompt-len", type=int, default=int(os.environ.get("NANOVLLM_BENCH_PROMPT_LEN", "128")))
    parser.add_argument("--gen-len", type=int, default=int(os.environ.get("NANOVLLM_BENCH_GEN_LEN", "32")))
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("NANOVLLM_BENCH_REPEAT", "3")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("NANOVLLM_BENCH_WARMUP", "1")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("NANOVLLM_TEMPERATURE", "0.0")))
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
        help="Disable native persistent graph bucket reuse for A/B measurements.",
    )
    parser.add_argument(
        "--native-vulkan-graph-reuse",
        action="store_true",
        default=os.environ.get("NANOVLLM_NATIVE_VULKAN_GRAPH_REUSE", "0") == "1",
        help="Experimental: reuse stable single-sequence Vulkan decode/MTP graph buckets.",
    )
    parser.add_argument(
        "--native-attention-impl",
        choices=("math", "auto", "flash", "paged"),
        default=os.environ.get("NANOVLLM_NATIVE_ATTENTION_IMPL", "auto"),
        help="Native attention implementation. 'paged' is an explicit Vulkan direct-KV experiment; auto remains unchanged.",
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
    parser.add_argument(
        "--native-mtp-verification-kv-fusion",
        action="store_true",
        default=os.environ.get("NANOVLLM_NATIVE_MTP_VERIFICATION_KV_FUSION", "0") == "1",
        help="Fuse target-conditioned MTP KV maintenance into each verification graph.",
    )
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
            native_mtp_verification_kv_fusion=args.native_mtp_verification_kv_fusion,
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
    return len(llm.tokenizer)


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
        result = llm.model_runner.call("run", plan)
        llm.scheduler.postprocess(seqs, result, is_prefill)
        llm.flush_backend_releases()
        elapsed = perf_counter() - t0

        retained_tokens = sum(
            seq.num_completion_tokens - before
            for seq, before in zip(seqs, completion_counts)
        )
        generated_tokens += retained_tokens
        if is_prefill:
            prefill_s += elapsed
            prefill_tokens += scheduled_tokens
        else:
            decode_s += elapsed
            # MTP may return more verified tokens than the request can still
            # retain. Count only the completion delta accepted by Scheduler so
            # the reported decode throughput is not inflated by that overrun.
            decode_tokens += retained_tokens

    return {
        "total_s": perf_counter() - total_start,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "generated_tokens": generated_tokens,
    }


def benchmark(args: argparse.Namespace) -> BenchResult:
    if (
        args.batch_size <= 0
        or args.prompt_len <= 0
        or args.gen_len <= 0
        or args.repeat <= 0
        or args.warmup < 0
    ):
        raise SystemExit(
            "--batch-size, --prompt-len, --gen-len and --repeat must be positive; "
            "--warmup must be non-negative."
        )
    if args.batch_size > args.max_num_seqs:
        raise SystemExit("--batch-size must be <= --max-num-seqs.")
    if args.prompt_len + args.gen_len > args.max_model_len:
        raise SystemExit("--prompt-len + --gen-len must be <= --max-model-len.")
    if args.backend != "cuda" and args.temperature != 0:
        raise SystemExit("CPU/Vulkan GGUF backends currently require --temperature 0.")
    if args.backend != "cuda" and args.use_prefix_cache:
        raise SystemExit("--use-prefix-cache is currently supported only by the CUDA backend.")
    if args.backend == "cuda" and args.enable_mtp:
        raise SystemExit("--enable-mtp is supported only by CPU/Vulkan GGUF backends.")

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

    mtp_baseline = {
        "drafted_tokens": 0,
        "accepted_tokens": 0,
        "verification_steps": 0,
    }
    mtp_stats = {
        "drafted_tokens": 0,
        "accepted_tokens": 0,
        "verification_steps": 0,
    }
    # Native runtime timing is the only breakdown which includes synchronous
    # backend execution, graph setup and host/device transfers. Keep it next
    # to the end-to-end counters so a high MTP acceptance rate cannot hide an
    # expensive verification path (especially on Vulkan).
    mtp_profile_baseline: dict[str, int] = {}
    mtp_profile: dict[str, int] = {}
    graph_stats = {"hits": 0, "misses": 0, "evictions": 0, "active_entries": 0}
    actual_num_kvcache_blocks = int(llm.config.num_kvcache_blocks)
    actual_model = os.fspath(llm.config.gguf_model or llm.config.model)
    ggml_commit = ""
    vulkan_compiled = False
    cuda_compiled = False
    cuda_graphs_compiled = False
    if args.backend.startswith("native"):
        from nanovllm.backends.native import build_info

        native_info = build_info()
        ggml_commit = str(native_info["ggml_commit"])
        vulkan_compiled = bool(native_info["vulkan"])
        cuda_compiled = bool(native_info["cuda"])
        cuda_graphs_compiled = bool(native_info["cuda_graphs"])
    get_mtp_stats = getattr(llm.model_runner, "mtp_stats", None)
    get_mtp_profile = getattr(llm.model_runner, "mtp_profile_stats", None)
    get_graph_stats = getattr(llm.model_runner, "graph_reuse_stats", None)
    try:
        run_index = 0
        for _ in range(args.warmup):
            prompts = make_batch_prompts(prompt_ids, args.batch_size, vocab_size, run_index, args.use_prefix_cache)
            run_generation_once(llm, prompts, sampling_params)
            run_index += 1
        if callable(get_mtp_stats):
            mtp_baseline.update(get_mtp_stats())
        if callable(get_mtp_profile):
            mtp_profile_baseline.update(get_mtp_profile())

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
        if callable(get_mtp_stats):
            mtp_stats.update(get_mtp_stats())
        if callable(get_mtp_profile):
            mtp_profile.update(get_mtp_profile())
        if callable(get_graph_stats):
            graph_stats.update(get_graph_stats())
        llm.exit()

    prefill_tok_s = totals["prefill_tokens"] / totals["prefill_s"] if totals["prefill_s"] > 0 else 0.0
    decode_tok_s = totals["decode_tokens"] / totals["decode_s"] if totals["decode_s"] > 0 else 0.0
    processed_tokens = totals["prefill_tokens"] + totals["decode_tokens"]
    processed_tok_s = processed_tokens / totals["total_s"] if totals["total_s"] > 0 else 0.0
    generated_tok_s = totals["generated_tokens"] / totals["total_s"] if totals["total_s"] > 0 else 0.0
    drafted_tokens = int(mtp_stats["drafted_tokens"] - mtp_baseline["drafted_tokens"])
    accepted_tokens = int(mtp_stats["accepted_tokens"] - mtp_baseline["accepted_tokens"])
    acceptance_rate = accepted_tokens / drafted_tokens if drafted_tokens > 0 else 0.0
    def profile_delta(key: str) -> int:
        return int(mtp_profile.get(key, 0) - mtp_profile_baseline.get(key, 0))

    draft_tokens = profile_delta("draft_tokens")
    verification_tokens = profile_delta("verification_tokens")
    kv_update_tokens = profile_delta("kv_update_tokens")
    draft_s = profile_delta("draft_elapsed_ns") / 1_000_000_000.0
    verification_s = profile_delta("verification_elapsed_ns") / 1_000_000_000.0
    kv_update_s = profile_delta("kv_update_elapsed_ns") / 1_000_000_000.0
    return BenchResult(
        backend=args.backend,
        model=actual_model,
        batch_size=args.batch_size,
        prompt_len=len(prompt_ids),
        gen_len=args.gen_len,
        use_prefix_cache=args.use_prefix_cache,
        repeat=args.repeat,
        warmup=args.warmup,
        threads=args.threads,
        device_index=args.device_index,
        num_kvcache_blocks=actual_num_kvcache_blocks,
        enable_mtp=args.enable_mtp,
        mtp_max_draft_tokens=args.mtp_max_draft_tokens,
        mtp_drafted_tokens=drafted_tokens,
        mtp_accepted_tokens=accepted_tokens,
        mtp_verification_steps=int(
            mtp_stats["verification_steps"] - mtp_baseline["verification_steps"]
        ),
        mtp_acceptance_rate=acceptance_rate,
        mtp_draft_s=draft_s,
        mtp_verification_s=verification_s,
        mtp_kv_update_s=kv_update_s,
        mtp_draft_graph_setup_s=profile_delta("draft_graph_setup_elapsed_ns") / 1_000_000_000.0,
        mtp_verification_graph_setup_s=profile_delta("verification_graph_setup_elapsed_ns") / 1_000_000_000.0,
        mtp_kv_update_graph_setup_s=profile_delta("kv_update_graph_setup_elapsed_ns") / 1_000_000_000.0,
        mtp_draft_tok_s=draft_tokens / draft_s if draft_s > 0 else 0.0,
        mtp_verification_tok_s=verification_tokens / verification_s if verification_s > 0 else 0.0,
        mtp_kv_update_tok_s=kv_update_tokens / kv_update_s if kv_update_s > 0 else 0.0,
        enable_graph_reuse=not args.no_graph_reuse,
        native_vulkan_graph_reuse=args.native_vulkan_graph_reuse,
        native_attention_impl=args.native_attention_impl,
        native_batched_recurrent_snapshots=args.native_batched_recurrent_snapshots,
        native_mtp_prefill_fusion=args.native_mtp_prefill_fusion,
        native_mtp_verification_kv_fusion=args.native_mtp_verification_kv_fusion,
        graph_cache_hits=int(graph_stats["hits"]),
        graph_cache_misses=int(graph_stats["misses"]),
        graph_cache_evictions=int(graph_stats["evictions"]),
        graph_cache_active_entries=int(graph_stats["active_entries"]),
        ggml_commit=ggml_commit,
        vulkan_compiled=vulkan_compiled,
        cuda_compiled=cuda_compiled,
        cuda_graphs_compiled=cuda_graphs_compiled,
        load_s=load_s,
        load_rss_mib=load_rss_mib,
        total_s=totals["total_s"],
        prefill_s=totals["prefill_s"],
        decode_s=totals["decode_s"],
        prefill_tokens=totals["prefill_tokens"],
        decode_tokens=totals["decode_tokens"],
        processed_tokens=processed_tokens,
        generated_tokens=totals["generated_tokens"],
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        processed_tok_s=processed_tok_s,
        generated_tok_s=generated_tok_s,
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
    print(f"warmup:             {result.warmup}")
    print(f"threads:            {result.threads}")
    print(f"device index:       {result.device_index}")
    print(f"KV blocks:          {result.num_kvcache_blocks}")
    print(f"MTP:                {result.enable_mtp} (K={result.mtp_max_draft_tokens})")
    print(f"graph reuse:        {result.enable_graph_reuse}")
    print(f"Vulkan graph reuse: {result.native_vulkan_graph_reuse}")
    print(f"attention impl:     {result.native_attention_impl}")
    print(f"batched snapshots:  {result.native_batched_recurrent_snapshots}")
    print(f"MTP prefill fusion: {result.native_mtp_prefill_fusion}")
    print(f"MTP verification KV fusion: {result.native_mtp_verification_kv_fusion}")
    if result.ggml_commit:
        print(f"GGML commit:        {result.ggml_commit}")
        print(f"Vulkan compiled:    {result.vulkan_compiled}")
        print(f"CUDA compiled:      {result.cuda_compiled}")
        print(f"CUDA Graphs:        {result.cuda_graphs_compiled}")
    print(f"load time:          {result.load_s:.3f} s")
    if result.load_rss_mib > 0:
        print(f"load RSS:           {result.load_rss_mib:.2f} MiB")
    print()
    print("metric              tokens        time(s)      tok/s")
    print(f"prefill             {result.prefill_tokens:>8}  {result.prefill_s:>10.3f}  {result.prefill_tok_s:>9.2f}")
    print(f"decode steps        {result.decode_tokens:>8}  {result.decode_s:>10.3f}  {result.decode_tok_s:>9.2f}")
    print(f"processed total     {result.processed_tokens:>8}  {result.total_s:>10.3f}  {result.processed_tok_s:>9.2f}")
    print(f"generated total     {result.generated_tokens:>8}  {result.total_s:>10.3f}  {result.generated_tok_s:>9.2f}")
    if result.enable_mtp:
        print(
            "MTP drafted/accepted/verifications: "
            f"{result.mtp_drafted_tokens}/{result.mtp_accepted_tokens}/"
            f"{result.mtp_verification_steps}; acceptance={result.mtp_acceptance_rate:.2%}"
        )
        print(
            "MTP stage wall(s) draft/verify/kv-update: "
            f"{result.mtp_draft_s:.3f}/{result.mtp_verification_s:.3f}/"
            f"{result.mtp_kv_update_s:.3f}"
        )
        print(
            "MTP graph setup(s) draft/verify/kv-update: "
            f"{result.mtp_draft_graph_setup_s:.3f}/"
            f"{result.mtp_verification_graph_setup_s:.3f}/"
            f"{result.mtp_kv_update_graph_setup_s:.3f}"
        )
    print(
        "graph cache hits/misses/evictions/active: "
        f"{result.graph_cache_hits}/{result.graph_cache_misses}/"
        f"{result.graph_cache_evictions}/{result.graph_cache_active_entries}"
    )


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
