"""Opt-in long-context greedy gates for the supported native runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


MODEL_ENV = "NANOVLLM_TEST_QWEN35_GGUF"
BACKENDS_ENV = "NANOVLLM_TEST_QWEN35_LONG_BACKENDS"
RESULT_PREFIX = "NANOVLLM_LONG_PROBE_JSON="
PROMPT = [9419, 1814] * 260 + [9419]
GENERATED_TOKENS = 31


def _run_probe(args: argparse.Namespace) -> None:
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        backend=f"native_{args.backend}",
        gguf_model=args.model,
        max_model_len=len(PROMPT) + GENERATED_TOKENS,
        max_num_batched_tokens=args.max_batch,
        max_num_seqs=1,
        num_kvcache_blocks=3,
        enable_mtp=args.mtp,
        mtp_max_draft_tokens=3,
        device_config={"n_threads": args.threads, "device_index": 0},
    )
    try:
        output = llm.generate(
            [PROMPT],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=GENERATED_TOKENS),
            use_tqdm=False,
        )[0]
        print(RESULT_PREFIX + json.dumps({
            "backend": args.backend,
            "max_batch": args.max_batch,
            "mtp": args.mtp,
            "token_ids": list(output["token_ids"]),
            "mtp_stats": llm.model_runner.mtp_stats() if args.mtp else None,
        }, sort_keys=True))
    finally:
        llm.exit()


class TestNativeQwen35LongContextOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(MODEL_ENV)
        requested = os.environ.get(BACKENDS_ENV)
        if not cls.model or not requested:
            raise unittest.SkipTest(
                f"set {MODEL_ENV} and {BACKENDS_ENV}=cpu,vulkan to run long-context gates"
            )
        if not Path(cls.model).is_file():
            raise AssertionError(f"GGUF does not exist: {cls.model}")
        cls.backends = tuple(part.strip() for part in requested.split(",") if part.strip())
        if not cls.backends or set(cls.backends) - {"cpu", "vulkan", "cuda"}:
            raise AssertionError(f"{BACKENDS_ENV} must list native cpu, vulkan and/or cuda")
        cls.threads = int(os.environ.get("NANOVLLM_TEST_CPU_THREADS", "8"))

    @classmethod
    def probe(cls, backend: str, max_batch: int, mtp: bool) -> dict:
        command = [
            sys.executable, os.fspath(Path(__file__).resolve()), "--probe",
            "--model", cls.model, "--backend", backend,
            "--max-batch", str(max_batch), "--threads", str(cls.threads),
        ]
        if mtp:
            command.append("--mtp")
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(RESULT_PREFIX):
                return json.loads(line.removeprefix(RESULT_PREFIX))
        raise AssertionError(f"probe produced no result:\n{completed.stdout}\n{completed.stderr}")

    def test_chunk_boundary_and_mtp_preserve_greedy_trace(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                baseline = self.probe(backend, 768, False)["token_ids"]
                self.assertEqual(len(baseline), GENERATED_TOKENS)
                for max_batch, mtp in ((64, False), (768, True), (64, True)):
                    result = self.probe(backend, max_batch, mtp)
                    self.assertEqual(result["token_ids"], baseline)
                    if mtp:
                        stats = result["mtp_stats"]
                        self.assertGreater(stats["drafted_tokens"], 0)
                        self.assertGreater(stats["verification_steps"], 0)
                        self.assertLessEqual(stats["accepted_tokens"], stats["drafted_tokens"])


def _parse_probe_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("cpu", "vulkan", "cuda"), required=True)
    # The unittest gate uses 64 and 768, while the standalone probe also
    # accepts intermediate sizes for backend correctness triage.
    parser.add_argument("--max-batch", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--mtp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    if "--probe" in sys.argv:
        _run_probe(_parse_probe_args())
    else:
        unittest.main()
