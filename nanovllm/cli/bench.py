"""Reproducible single-sequence benchmark for the public native runtime."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from time import perf_counter

from nanovllm import LLM, SamplingParams
from nanovllm.backends.base import build_execution_plan


@dataclass(slots=True)
class BenchResult:
    backend: str
    model: str
    prompt_len: int
    gen_len: int
    repeat: int
    warmup: int
    threads: int
    device_index: int
    num_kvcache_blocks: int
    enable_mtp: bool
    mtp_max_draft_tokens: int
    mali_experimental_graph_reuse: bool
    mtp_drafted_tokens: int
    mtp_accepted_tokens: int
    mtp_verification_steps: int
    mtp_acceptance_rate: float
    graph_cache_hits: int
    graph_cache_misses: int
    graph_cache_evictions: int
    graph_cache_active_entries: int
    ggml_commit: str
    vulkan_compiled: bool
    cuda_compiled: bool
    cuda_graphs_compiled: bool
    load_s: float
    total_s: float
    prefill_s: float
    decode_s: float
    prefill_tokens: int
    decode_tokens: int
    generated_tokens: int
    prefill_tok_s: float
    decode_tok_s: float
    processed_tok_s: float
    generated_tok_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanovllm-ggml native single-sequence benchmark")
    parser.add_argument("model", nargs="?", default=None, help="Qwen3.5 GGUF model path")
    parser.add_argument(
        "--backend",
        choices=("native_cpu", "native_vulkan", "native_cuda"),
        default=os.environ.get("NANOVLLM_BACKEND", "native_cpu"),
    )
    parser.add_argument("--gguf-model", default=os.environ.get("NANOVLLM_GGUF_MODEL"))
    parser.add_argument("--max-model-len", type=int, default=8092)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("NANOVLLM_THREADS", "8")))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt", default="Benchmark prompt for nanovllm-ggml. ")
    parser.add_argument("--enable-mtp", action="store_true")
    parser.add_argument("--mtp-max-draft-tokens", type=int, default=1)
    parser.add_argument(
        "--mali-experimental-graph-reuse",
        action="store_true",
        help="Experimental: enable persistent graph reuse on Mali Vulkan.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _build_llm(args: argparse.Namespace) -> LLM:
    model = args.gguf_model or args.model
    if not model:
        raise SystemExit("Please pass a GGUF model path or set NANOVLLM_GGUF_MODEL.")
    return LLM(
        model,
        backend=args.backend,
        gguf_model=model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enable_mtp=args.enable_mtp,
        mtp_max_draft_tokens=args.mtp_max_draft_tokens,
        mali_experimental_graph_reuse=args.mali_experimental_graph_reuse,
        device_config={"n_threads": args.threads, "device_index": args.device_index},
    )


def _prompt_tokens(llm: LLM, text: str, length: int) -> list[int]:
    token_ids = list(llm.model_runner.call("tokenize", text))
    if not token_ids:
        raise SystemExit("benchmark prompt produced no token IDs")
    return (token_ids * ((length + len(token_ids) - 1) // len(token_ids)))[:length]


def _run_once(llm: LLM, prompt: list[int], gen_len: int) -> dict[str, float | int]:
    llm.add_request(prompt, SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=gen_len))
    totals: dict[str, float | int] = {
        "total_s": 0.0, "prefill_s": 0.0, "decode_s": 0.0,
        "prefill_tokens": 0, "decode_tokens": 0, "generated_tokens": 0,
    }
    started = perf_counter()
    while not llm.is_finished():
        step_started = perf_counter()
        seqs, is_prefill = llm.scheduler.schedule()
        llm.flush_backend_releases()
        plan = build_execution_plan(seqs, is_prefill, llm.config.kvcache_block_size)
        scheduled = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)
        completed_before = sum(seq.num_completion_tokens for seq in seqs)
        result = llm.model_runner.call("run", plan)
        llm.scheduler.postprocess(seqs, result, is_prefill)
        llm.flush_backend_releases()
        elapsed = perf_counter() - step_started
        completed = sum(seq.num_completion_tokens for seq in seqs) - completed_before
        if is_prefill:
            totals["prefill_s"] += elapsed
            totals["prefill_tokens"] += scheduled
        else:
            totals["decode_s"] += elapsed
            totals["decode_tokens"] += completed
        totals["generated_tokens"] += completed
    totals["total_s"] = perf_counter() - started
    return totals


def benchmark(args: argparse.Namespace) -> BenchResult:
    if min(args.prompt_len, args.gen_len, args.repeat) <= 0 or args.warmup < 0:
        raise SystemExit("prompt/gen/repeat must be positive and warmup non-negative")
    if args.prompt_len + args.gen_len > args.max_model_len:
        raise SystemExit("--prompt-len + --gen-len must be <= --max-model-len")
    if args.temperature != 0:
        raise SystemExit("native runtime currently supports --temperature 0 only")

    load_started = perf_counter()
    llm = _build_llm(args)
    load_s = perf_counter() - load_started
    prompt = _prompt_tokens(llm, args.prompt, args.prompt_len)
    get_mtp = getattr(llm.model_runner, "mtp_stats", lambda: {})
    get_graph = getattr(llm.model_runner, "graph_reuse_stats", lambda: {})
    mtp_before = dict(get_mtp())
    try:
        for _ in range(args.warmup):
            _run_once(llm, prompt, args.gen_len)
        mtp_before = dict(get_mtp())
        totals = {key: 0.0 for key in ("total_s", "prefill_s", "decode_s")}
        totals.update({key: 0 for key in ("prefill_tokens", "decode_tokens", "generated_tokens")})
        for _ in range(args.repeat):
            result = _run_once(llm, prompt, args.gen_len)
            for key, value in result.items():
                totals[key] += value
        mtp_after = dict(get_mtp())
        graph = dict(get_graph())
    finally:
        llm.exit()

    from nanovllm.backends.native import build_info
    info = build_info()
    drafted = int(mtp_after.get("drafted_tokens", 0) - mtp_before.get("drafted_tokens", 0))
    accepted = int(mtp_after.get("accepted_tokens", 0) - mtp_before.get("accepted_tokens", 0))
    verification = int(mtp_after.get("verification_steps", 0) - mtp_before.get("verification_steps", 0))
    prefill_s, decode_s, total_s = (float(totals[key]) for key in ("prefill_s", "decode_s", "total_s"))
    prefill_tokens, decode_tokens, generated = (int(totals[key]) for key in ("prefill_tokens", "decode_tokens", "generated_tokens"))
    return BenchResult(
        backend=args.backend, model=os.fspath(args.gguf_model or args.model),
        prompt_len=args.prompt_len, gen_len=args.gen_len, repeat=args.repeat, warmup=args.warmup,
        threads=args.threads, device_index=args.device_index,
        num_kvcache_blocks=int(llm.config.num_kvcache_blocks), enable_mtp=args.enable_mtp,
        mtp_max_draft_tokens=args.mtp_max_draft_tokens,
        mali_experimental_graph_reuse=args.mali_experimental_graph_reuse,
        mtp_drafted_tokens=drafted, mtp_accepted_tokens=accepted,
        mtp_verification_steps=verification,
        mtp_acceptance_rate=accepted / drafted if drafted else 0.0,
        graph_cache_hits=int(graph.get("hits", 0)), graph_cache_misses=int(graph.get("misses", 0)),
        graph_cache_evictions=int(graph.get("evictions", 0)), graph_cache_active_entries=int(graph.get("active_entries", 0)),
        ggml_commit=str(info["ggml_commit"]), vulkan_compiled=bool(info["vulkan"]),
        cuda_compiled=bool(info["cuda"]), cuda_graphs_compiled=bool(info["cuda_graphs"]),
        load_s=load_s, total_s=total_s, prefill_s=prefill_s, decode_s=decode_s,
        prefill_tokens=prefill_tokens, decode_tokens=decode_tokens, generated_tokens=generated,
        prefill_tok_s=prefill_tokens / prefill_s if prefill_s else 0.0,
        decode_tok_s=decode_tokens / decode_s if decode_s else 0.0,
        processed_tok_s=(prefill_tokens + decode_tokens) / total_s if total_s else 0.0,
        generated_tok_s=generated / total_s if total_s else 0.0,
    )


def main() -> int:
    args = parse_args()
    result = benchmark(args)
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        for key, value in asdict(result).items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
