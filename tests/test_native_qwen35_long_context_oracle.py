"""Opt-in long-context correctness gates for the native Qwen3.5 runtime.

Every configuration runs in a fresh process.  Besides matching production's
usual one-runtime lifecycle, process isolation prevents a failing Vulkan
backend from contaminating the remaining cells of the comparison matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


MODEL_ENV = "NANOVLLM_TEST_QWEN35_GGUF"
TOKENIZER_ENV = "NANOVLLM_TEST_QWEN35_TOKENIZER"
BACKENDS_ENV = "NANOVLLM_TEST_QWEN35_LONG_BACKENDS"
ATTENTION_IMPL_ENV = "NANOVLLM_TEST_QWEN35_ATTENTION_IMPL"
MALI_MTP_PREFILL_STRATEGIES_ENV = "NANOVLLM_TEST_QWEN35_MALI_MTP_PREFILL_STRATEGIES"
RESULT_PREFIX = "NANOVLLM_LONG_PROBE_JSON="
PROMPT = [9419, 1814] * 260 + [9419]
GENERATED_TOKENS = 31


def _run_probe(args: argparse.Namespace) -> None:
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        backend=f"native_{args.backend}",
        model_format="gguf",
        gguf_model=args.model,
        tokenizer=args.tokenizer,
        tokenizer_backend="hf",
        max_model_len=len(PROMPT) + GENERATED_TOKENS,
        max_num_batched_tokens=args.max_batch,
        max_num_seqs=1,
        num_kvcache_blocks=3,
        enable_mtp=args.mtp,
        mtp_max_draft_tokens=3,
        device_config={"n_threads": args.threads, "device_index": 0},
        native_attention_impl=args.attention_impl,
        native_mtp_prefill_fusion=args.mtp_prefill_fusion,
        native_mtp_verification_kv_fusion=args.mtp_verification_kv_fusion,
        native_mali_mtp_prefill_strategy=args.mali_mtp_prefill_strategy,
    )
    try:
        output = llm.generate(
            [PROMPT],
            SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=GENERATED_TOKENS,
            ),
            use_tqdm=False,
        )[0]
        stats = llm.model_runner.mtp_stats() if args.mtp else None
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "backend": args.backend,
                    "max_batch": args.max_batch,
                    "mtp": args.mtp,
                    "mtp_prefill_fusion": args.mtp_prefill_fusion,
                    "mtp_verification_kv_fusion": args.mtp_verification_kv_fusion,
                    "mali_mtp_prefill_strategy": args.mali_mtp_prefill_strategy,
                    "token_ids": list(output["token_ids"]),
                    "mtp_stats": stats,
                },
                sort_keys=True,
            )
        )
    finally:
        llm.exit()


class TestNativeQwen35LongContextOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(MODEL_ENV)
        cls.tokenizer = os.environ.get(TOKENIZER_ENV)
        requested = os.environ.get(BACKENDS_ENV)
        if not cls.model or not cls.tokenizer or not requested:
            raise unittest.SkipTest(
                f"set {MODEL_ENV}, {TOKENIZER_ENV}, and {BACKENDS_ENV}=cpu,vulkan "
                "to run long-context native gates"
            )

        if not Path(cls.model).is_file():
            raise AssertionError(f"GGUF does not exist: {cls.model}")
        if not Path(cls.tokenizer).exists():
            raise AssertionError(f"tokenizer path does not exist: {cls.tokenizer}")

        cls.backends = tuple(
            backend.strip() for backend in requested.split(",") if backend.strip()
        )
        invalid = sorted(set(cls.backends) - {"cpu", "vulkan"})
        if invalid or not cls.backends:
            raise AssertionError(
                f"{BACKENDS_ENV} must contain cpu and/or vulkan; invalid={invalid}"
            )
        try:
            cls.threads = int(os.environ.get("NANOVLLM_TEST_CPU_THREADS", "8"))
        except ValueError as exc:
            raise AssertionError("NANOVLLM_TEST_CPU_THREADS must be an integer") from exc
        if cls.threads <= 0:
            raise AssertionError("NANOVLLM_TEST_CPU_THREADS must be positive")
        cls.attention_impl = os.environ.get(ATTENTION_IMPL_ENV, "auto")
        if cls.attention_impl not in {"auto", "math", "flash", "paged"}:
            raise AssertionError(
                f"{ATTENTION_IMPL_ENV} must be auto, math, flash, or paged"
            )
        cls.mali_mtp_prefill_strategies = tuple(
            strategy.strip()
            for strategy in os.environ.get(
                MALI_MTP_PREFILL_STRATEGIES_ENV, "whole"
            ).split(",")
            if strategy.strip()
        )
        invalid_strategies = sorted(
            set(cls.mali_mtp_prefill_strategies)
            - {"whole", "chunked_legacy", "chunked_staged"}
        )
        if (
            invalid_strategies
            or not cls.mali_mtp_prefill_strategies
            or "whole" not in cls.mali_mtp_prefill_strategies
        ):
            raise AssertionError(
                f"{MALI_MTP_PREFILL_STRATEGIES_ENV} must include whole and may include "
                f"chunked_legacy/chunked_staged; invalid={invalid_strategies}"
            )

    @classmethod
    def probe(
        cls,
        backend: str,
        max_batch: int,
        mtp: bool,
        mtp_verification_kv_fusion: bool = False,
        mali_mtp_prefill_strategy: str = "whole",
    ) -> dict:
        command = [
            sys.executable,
            os.fspath(Path(__file__).resolve()),
            "--probe",
            "--model",
            cls.model,
            "--tokenizer",
            cls.tokenizer,
            "--backend",
            backend,
            "--max-batch",
            str(max_batch),
            "--threads",
            str(cls.threads),
            "--attention-impl",
            cls.attention_impl,
            "--mali-mtp-prefill-strategy",
            mali_mtp_prefill_strategy,
        ]
        if mtp:
            command.append("--mtp")
        if mtp_verification_kv_fusion:
            command.append("--mtp-verification-kv-fusion")
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(RESULT_PREFIX):
                return json.loads(line.removeprefix(RESULT_PREFIX))
        raise AssertionError(
            "long-context probe produced no result line:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    def test_mtp_and_chunk_boundaries_preserve_backend_greedy_trace(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                strategies = (
                    self.mali_mtp_prefill_strategies
                    if backend == "vulkan"
                    else ("whole",)
                )
                results = {}
                for max_batch in (768, 64):
                    for mtp in (False, True):
                        for fusion in ((False, True) if mtp else (False,)):
                            for strategy in strategies:
                                # Chunked Mali prefill is meaningful only with
                                # MTP enabled.  Cover both the legacy host-side
                                # KV update and the in-graph fused transaction:
                                # the latter is a candidate fix for the
                                # cross-graph chunk-boundary corruption.
                                if strategy != "whole" and not mtp:
                                    continue
                                key = (max_batch, mtp, fusion, strategy)
                                results[key] = self.probe(
                                    backend,
                                    max_batch,
                                    mtp,
                                    fusion,
                                    strategy,
                                )
                expected = results[(768, False, False, "whole")]["token_ids"]
                self.assertEqual(len(expected), GENERATED_TOKENS)
                for key, result in results.items():
                    self.assertEqual(
                        result["token_ids"],
                        expected,
                        f"{backend} max_batch={key[0]} mtp={key[1]} "
                        f"mtp_verification_kv_fusion={key[2]} "
                        f"mali_mtp_prefill_strategy={key[3]} changed the greedy trace",
                    )
                    if key[1]:
                        stats = result["mtp_stats"]
                        self.assertGreater(stats["drafted_tokens"], 0)
                        self.assertGreater(stats["verification_steps"], 0)
                        self.assertGreaterEqual(stats["accepted_tokens"], 0)
                        self.assertLessEqual(
                            stats["accepted_tokens"], stats["drafted_tokens"]
                        )


def _parse_probe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), required=True)
    parser.add_argument("--max-batch", choices=(64, 768), type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--attention-impl", choices=("auto", "math", "flash", "paged"), required=True
    )
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--mtp-prefill-fusion", action="store_true")
    parser.add_argument("--mtp-verification-kv-fusion", action="store_true")
    parser.add_argument(
        "--mali-mtp-prefill-strategy",
        choices=("whole", "chunked_legacy", "chunked_staged"),
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    if "--probe" in sys.argv:
        _run_probe(_parse_probe_args())
    else:
        unittest.main()
