"""Real-model CPU gates for the in-tree Qwen3.5 runtime.

The expected token traces below were captured from the staged official
llama.cpp oracle at commit 91c631b21d6e5d09e9c6659efdf6baeef5a44ddb,
using the exact Qwen3.5-2B Q4_0 GGUF contract accepted by ``nanovllm._C``.

These tests are deliberately opt-in because they load the 1.2 GiB GGUF and
execute the full model on CPU.  They do not use the staged backend at test
runtime, so the native-runtime gate remains independent of an external
``libllama`` installation.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from nanovllm import LLM, SamplingParams
from nanovllm.backends.base import BackendExecutionPlan
from nanovllm.backends.native.runner import NativeRunner


MODEL_ENV = "NANOVLLM_TEST_QWEN35_GGUF"
TOKENIZER_ENV = "NANOVLLM_TEST_QWEN35_TOKENIZER"
BACKEND_ENV = "NANOVLLM_TEST_QWEN35_BACKEND"
SEQUENCE_ID = 701
BLOCK_SIZE = 256


class TestNativeQwen35CpuOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_value = os.environ.get(MODEL_ENV)
        tokenizer_value = os.environ.get(TOKENIZER_ENV)
        if not model_value or not tokenizer_value:
            raise unittest.SkipTest(
                f"set both {MODEL_ENV} and {TOKENIZER_ENV} to run real-model CPU gates"
            )

        cls.model_path = Path(model_value)
        cls.tokenizer_path = Path(tokenizer_value)
        if not cls.model_path.is_file():
            raise unittest.SkipTest(f"GGUF does not exist: {cls.model_path}")
        if not cls.tokenizer_path.exists():
            raise unittest.SkipTest(f"HF tokenizer path does not exist: {cls.tokenizer_path}")

        try:
            from nanovllm import _C
        except ImportError as exc:
            raise unittest.SkipTest(f"native extension is not built: {exc}") from exc
        if not hasattr(_C, "Qwen35Runtime"):
            raise unittest.SkipTest(
                "native extension does not expose Qwen35Runtime; rebuild it first"
            )

        contract = _C.qwen35_model_info(path=os.fspath(cls.model_path))
        if (
            contract["architecture"] != "qwen35"
            or contract["main_layers"] != 24
            or contract["nextn_predict_layers"] != 1
            or not contract["strict_q4_0_profile"]
        ):
            raise AssertionError(
                "CPU oracle gates require the exact Qwen3.5-2B Q4_0 + bundled-MTP contract"
            )

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise unittest.SkipTest(f"transformers tokenizer support is unavailable: {exc}") from exc
        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.fspath(cls.tokenizer_path), use_fast=True
        )

        try:
            cls.cpu_threads = int(os.environ.get("NANOVLLM_TEST_CPU_THREADS", "4"))
        except ValueError as exc:
            raise AssertionError("NANOVLLM_TEST_CPU_THREADS must be an integer") from exc
        if cls.cpu_threads <= 0:
            raise AssertionError("NANOVLLM_TEST_CPU_THREADS must be positive")
        cls.backend = os.environ.get(BACKEND_ENV, "cpu")
        if cls.backend not in {"cpu", "vulkan"}:
            raise AssertionError(f"{BACKEND_ENV} must be cpu or vulkan")

    def make_runner(
        self,
        *,
        enable_mtp: bool,
        max_model_len: int = BLOCK_SIZE,
    ) -> NativeRunner:
        config = SimpleNamespace(
            model=os.fspath(self.model_path),
            gguf_model=os.fspath(self.model_path),
            backend=f"native_{self.backend}",
            device_config={"n_threads": self.cpu_threads, "device_index": 0},
            max_model_len=max_model_len,
            max_num_batched_tokens=16,
            max_num_seqs=1,
            kvcache_block_size=BLOCK_SIZE,
            num_kvcache_blocks=1,
            enable_mtp=enable_mtp,
            mtp_max_draft_tokens=3,
        )
        return NativeRunner(config)

    @staticmethod
    def plan(tokens: list[int], *, start: int, mode: str) -> BackendExecutionPlan:
        count = len(tokens)
        return BackendExecutionPlan(
            mode=mode,
            input_ids=list(tokens),
            positions=list(range(start, start + count)),
            seq_ids=[SEQUENCE_ID],
            scheduled_token_counts=[count],
            block_tables=[[0]],
            slot_mapping=list(range(start, start + count)),
            context_lens=[start + count],
            num_cached_tokens=[start],
            temperatures=[0.0],
        )

    def prompt_tokens(self, prompt: str, expected: list[int]) -> list[int]:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        self.assertEqual(tokens, expected, "HF tokenizer does not match the oracle tokenizer")
        return tokens

    def prefill(self, runner: NativeRunner, prompt_ids: list[int]) -> int:
        result = runner.run(self.plan(prompt_ids, start=0, mode="prefill"))
        self.assertEqual(result.draft_token_counts, None)
        self.assertEqual(len(result.token_ids), 1)
        self.assertEqual(len(result.token_ids[0]), 1)
        return result.token_ids[0][0]

    def decode_target(self, runner: NativeRunner, pending: int, position: int) -> int:
        result = runner.run(self.plan([pending], start=position, mode="decode"))
        self.assertEqual(result.draft_token_counts, None)
        self.assertEqual(len(result.token_ids), 1)
        self.assertEqual(len(result.token_ids[0]), 1)
        return result.token_ids[0][0]

    def mtp_trace(
        self,
        runner: NativeRunner,
        *,
        prompt_ids: list[int],
    ) -> tuple[int, list[int], list[int]]:
        """Prefill and run two K=3 verification steps, then release the slot."""

        pending = self.prefill(runner, prompt_ids)
        position = len(prompt_ids)
        first = runner.run(self.plan([pending], start=position, mode="decode"))
        self.assertEqual(first.draft_token_counts, [3])
        position += len(first.token_ids[0])
        pending = first.token_ids[0][-1]
        second = runner.run(self.plan([pending], start=position, mode="decode"))
        self.assertEqual(second.draft_token_counts, [3])

        # Release both Paged-KV block 0 and the native recurrent-state slot.
        # Every scenario below reuses the same external sequence ID and block,
        # making stale cache/state observable as an oracle mismatch.
        runner.release_blocks([0], [SEQUENCE_ID])
        self.assertEqual(runner._seq_slots, {})
        return pending, first.token_ids[0], second.token_ids[0]

    def test_target_two_decode_steps_match_official_cpu_oracle(self):
        runner = self.make_runner(enable_mtp=False)
        try:
            prompt_ids = self.prompt_tokens("Hello", [9419])
            first_pending = self.prefill(runner, prompt_ids)
            first_decode = self.decode_target(runner, first_pending, len(prompt_ids))
            second_decode = self.decode_target(
                runner, first_decode, len(prompt_ids) + 1
            )

            # Official llama.cpp greedy trace: ", I am".  This prompt has a
            # stable argmax margin across CPU_REPACK and ordinary Q4_0 kernels;
            # near-tied logits are intentionally not used as token oracles.
            self.assertEqual([first_pending, first_decode, second_decode], [11, 353, 1044])
            runner.release_blocks([0], [SEQUENCE_ID])
        finally:
            runner.shutdown()

    def test_mtp_k3_accept_reject_rollback_release_and_shutdown(self):
        runner = self.make_runner(enable_mtp=True)
        try:
            # Full acceptance: a=3, so rollback selects newest snapshot plane 0.
            full_ids = self.prompt_tokens(
                "Once upon a time", [12162, 5028, 264, 854]
            )
            _, first, second = self.mtp_trace(
                runner, prompt_ids=full_ids
            )
            self.assertEqual(first, [303, 264, 22960, 1814])
            self.assertEqual(second, [11, 1017, 557, 264])
            self.assertEqual(len(first) - 1, 3)

            # Partial acceptance: a=1 selects plane K-a=2.  The following
            # verification step is the observable rollback/catch-up check.
            partial_ids = self.prompt_tokens(
                "The capital of France is", [760, 6511, 314, 9338, 369]
            )
            _, first, second = self.mtp_trace(
                runner, prompt_ids=partial_ids
            )
            self.assertEqual(first, [13, 561])
            self.assertEqual(second, [6511, 314, 279])
            self.assertEqual(len(first) - 1, 1)

            # Immediate rejection: a=0 selects oldest plane K=3.  Its second
            # step verifies that recurrent and MTP KV state were rolled back.
            reject_ids = self.prompt_tokens("Hello", [9419])
            _, first, second = self.mtp_trace(
                runner, prompt_ids=reject_ids
            )
            self.assertEqual(first, [353])
            self.assertEqual(second, [1044])
            self.assertEqual(len(first) - 1, 0)

            self.assertEqual(
                runner.mtp_stats(),
                {
                    "drafted_tokens": 18,
                    "accepted_tokens": 9,
                    "verification_steps": 6,
                },
            )
        finally:
            runner.shutdown()

        # shutdown is idempotent, and no execution is permitted afterwards.
        runner.shutdown()
        with self.assertRaisesRegex(RuntimeError, "shut down"):
            runner.run(self.plan([11], start=0, mode="prefill"))

    def test_mtp_near_context_boundary_falls_back_to_target_greedy(self):
        runner = self.make_runner(enable_mtp=True, max_model_len=2)
        try:
            prompt_ids = self.prompt_tokens("Hello", [9419])
            pending = self.prefill(runner, prompt_ids)
            self.assertEqual(pending, 11)

            result = runner.run(self.plan([pending], start=1, mode="decode"))
            self.assertEqual(result.token_ids, [[353]])
            self.assertIsNone(result.draft_token_counts)
            self.assertEqual(
                runner.mtp_stats(),
                {
                    "drafted_tokens": 0,
                    "accepted_tokens": 0,
                    "verification_steps": 0,
                },
            )
            runner.release_blocks([0], [SEQUENCE_ID])
        finally:
            runner.shutdown()

    def test_full_nanovllm_scheduler_path_matches_with_mtp_off_and_on(self):
        expected = [0, 353, 1044, 264, 5286, 314]
        for enable_mtp in (False, True):
            with self.subTest(enable_mtp=enable_mtp):
                llm = LLM(
                    os.fspath(self.model_path),
                    backend="native_cpu",
                    gguf_model=os.fspath(self.model_path),
                    max_model_len=BLOCK_SIZE,
                    max_num_batched_tokens=16,
                    max_num_seqs=1,
                    num_kvcache_blocks=1,
                    enable_mtp=enable_mtp,
                    mtp_max_draft_tokens=3,
                    device_config={"n_threads": self.cpu_threads, "device_index": 0},
                )
                try:
                    self.assertTrue(llm.config.eos_token_ids)
                    self.assertTrue(
                        all(0 <= token < llm.model_runner.vocab_size
                            for token in llm.config.eos_token_ids)
                    )
                    outputs = llm.generate(
                        [[9419, 1814]],
                        SamplingParams(
                            temperature=0.0,
                            ignore_eos=True,
                            max_tokens=len(expected),
                        ),
                        use_tqdm=False,
                    )
                    self.assertEqual(outputs[0]["token_ids"], expected)
                    if enable_mtp:
                        stats = llm.model_runner.mtp_stats()
                        self.assertGreater(stats["drafted_tokens"], 0)
                        self.assertGreater(stats["verification_steps"], 0)
                        self.assertGreaterEqual(stats["accepted_tokens"], 0)
                        self.assertLessEqual(
                            stats["accepted_tokens"], stats["drafted_tokens"]
                        )
                finally:
                    llm.exit()


if __name__ == "__main__":
    unittest.main()
