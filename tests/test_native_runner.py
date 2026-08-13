import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from nanovllm.backends.base import BackendExecutionPlan
from nanovllm.backends.native.runner import NativeRunner


class FakeQwen35Runtime:
    instances = []

    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.run_calls = []
        self.mtp_calls = []
        self.release_calls = []
        self.shutdown_calls = 0
        self.greedy_token_ids = None
        self.mtp_result = None
        type(self).instances.append(self)

    def run(self, plan):
        self.run_calls.append(plan)
        if self.greedy_token_ids is not None:
            return self.greedy_token_ids
        return np.arange(plan["n_seqs"], dtype=np.int32) + 100

    def run_mtp(self, plan, capacity):
        self.mtp_calls.append((plan, capacity))
        if self.mtp_result is None:
            raise AssertionError("the test must configure an MTP result")
        return self.mtp_result

    def release_blocks(self, **kwargs):
        self.release_calls.append(kwargs)

    def eog_token_ids(self):
        return [248044, 248046]

    def tokenize(self, text):
        return [1, len(text), 2]

    def detokenize(self, token_ids):
        return ":".join(str(token) for token in token_ids)

    def shutdown(self):
        self.shutdown_calls += 1


def make_config(**overrides):
    values = {
        "model": "/models/Qwen3.5-2B-Q4_0.gguf",
        "gguf_model": "/models/Qwen3.5-2B-Q4_0.gguf",
        "backend": "native_cpu",
        "device_config": {"n_threads": 3, "device_index": 1, "library_path": "/do/not/use.so"},
        "max_model_len": 2048,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 2,
        "kvcache_block_size": 256,
        "num_kvcache_blocks": 8,
        "enable_mtp": False,
        "mtp_max_draft_tokens": 3,
        "enable_graph_reuse": True,
        "native_vulkan_graph_reuse": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_plan(
    *,
    mode="prefill",
    seq_ids=(42, 99),
    temperatures=None,
):
    seq_ids = list(seq_ids)
    if mode == "prefill":
        assert len(seq_ids) == 2
        return BackendExecutionPlan(
            mode="prefill",
            input_ids=[10, 11, 20],
            positions=[0, 1, 0],
            seq_ids=seq_ids,
            scheduled_token_counts=[2, 1],
            block_tables=[[3, 1], [2]],
            slot_mapping=[768, 769, 512],
            context_lens=[2, 1],
            num_cached_tokens=[0, 0],
            temperatures=list(temperatures if temperatures is not None else [0.0, 0.0]),
        )
    count = len(seq_ids)
    return BackendExecutionPlan(
        mode="decode",
        input_ids=[30 + index for index in range(count)],
        positions=[4] * count,
        seq_ids=seq_ids,
        scheduled_token_counts=[4] * count,
        block_tables=[[index + 1] for index in range(count)],
        slot_mapping=[(index + 1) * 256 + 4 for index in range(count)],
        context_lens=[5] * count,
        num_cached_tokens=[4] * count,
        temperatures=list(temperatures if temperatures is not None else [0.0] * count),
    )


class NativeRunnerTests(unittest.TestCase):

    def setUp(self):
        FakeQwen35Runtime.instances.clear()

    def make_runner(self, config=None, runtime_type=FakeQwen35Runtime):
        extension = ModuleType("nanovllm._C")
        extension.Qwen35Runtime = runtime_type
        with patch.dict(sys.modules, {"nanovllm._C": extension}):
            return NativeRunner(config or make_config())

    def test_missing_qwen35_runtime_type_has_an_actionable_error(self):
        extension = ModuleType("nanovllm._C")
        with patch.dict(sys.modules, {"nanovllm._C": extension}):
            with self.assertRaisesRegex(RuntimeError, "does not expose Qwen35Runtime"):
                NativeRunner(make_config())

    def test_constructor_forwards_only_the_in_tree_runtime_contract(self):
        runner = self.make_runner(make_config(backend="native_vulkan"))
        kwargs = runner.runtime.constructor_kwargs

        self.assertEqual(
            kwargs,
            {
                "model_path": "/models/Qwen3.5-2B-Q4_0.gguf",
                "backend": "vulkan",
                "max_model_len": 2048,
                "max_num_batched_tokens": 256,
                "max_num_seqs": 2,
                "block_size": 256,
                "num_blocks": 8,
                "n_threads": 3,
                "device_index": 1,
                "enable_mtp": False,
                "mtp_max_draft_tokens": 3,
                "enable_graph_reuse": True,
                "enable_vulkan_graph_reuse": False,
                "attention_impl": "auto",
                "enable_batched_recurrent_snapshots": False,
                "enable_mtp_prefill_fusion": False,
                "enable_mtp_verification_kv_fusion": False,
            },
        )
        self.assertNotIn("library_path", kwargs)

    def test_constructor_maps_native_cuda_to_the_ggml_cuda_runtime(self):
        runner = self.make_runner(make_config(backend="native_cuda"))
        self.assertEqual(runner.runtime.constructor_kwargs["backend"], "cuda")

    def test_constructor_forwards_native_execution_optimization_toggles(self):
        runner = self.make_runner(
            make_config(
                enable_mtp=True,
                native_attention_impl="auto",
                native_batched_recurrent_snapshots=True,
                native_mtp_prefill_fusion=True,
                native_mtp_verification_kv_fusion=True,
                native_vulkan_graph_reuse=True,
            )
        )
        kwargs = runner.runtime.constructor_kwargs
        self.assertEqual(kwargs["attention_impl"], "auto")
        self.assertTrue(kwargs["enable_batched_recurrent_snapshots"])
        self.assertTrue(kwargs["enable_mtp_prefill_fusion"])
        self.assertTrue(kwargs["enable_mtp_verification_kv_fusion"])
        self.assertTrue(kwargs["enable_vulkan_graph_reuse"])

    def test_greedy_run_flattens_and_converts_the_execution_plan(self):
        runner = self.make_runner()
        result = runner.run(make_plan())

        self.assertEqual(result.token_ids, [[100], [101]])
        self.assertIsNone(result.draft_token_counts)
        self.assertEqual(len(runner.runtime.run_calls), 1)
        plan = runner.runtime.run_calls[0]
        self.assertEqual(
            {key: plan[key] for key in ("is_prefill", "n_tokens", "n_seqs", "block_size", "block_table_cols")},
            {
                "is_prefill": True,
                "n_tokens": 3,
                "n_seqs": 2,
                "block_size": 256,
                "block_table_cols": 2,
            },
        )
        np.testing.assert_array_equal(plan["tokens"], [10, 11, 20])
        np.testing.assert_array_equal(plan["positions"], [0, 1, 0])
        np.testing.assert_array_equal(plan["seq_ids"], [0, 1])
        np.testing.assert_array_equal(plan["scheduled_token_counts"], [2, 1])
        np.testing.assert_array_equal(plan["slot_mapping"], [768, 769, 512])
        np.testing.assert_array_equal(plan["block_tables"], [3, 1, 2, -1])
        np.testing.assert_array_equal(plan["context_lens"], [2, 1])
        np.testing.assert_array_equal(plan["num_cached_tokens"], [0, 0])
        for key in (
            "tokens",
            "positions",
            "seq_ids",
            "scheduled_token_counts",
            "slot_mapping",
            "block_tables",
            "context_lens",
            "num_cached_tokens",
        ):
            self.assertEqual(plan[key].dtype, np.dtype(np.int32))
            self.assertTrue(plan[key].flags.c_contiguous)
            self.assertEqual(plan[key].ndim, 1)

    def test_nonzero_temperature_is_rejected_before_runtime_or_slot_changes(self):
        runner = self.make_runner()
        with self.assertRaisesRegex(ValueError, "only supports greedy"):
            runner.run(make_plan(temperatures=[0.0, 0.25]))
        self.assertEqual(runner.runtime.run_calls, [])
        self.assertEqual(runner._seq_slots, {})

    def test_mtp_returns_multiple_tokens_and_accumulates_stats(self):
        runner = self.make_runner(make_config(enable_mtp=True))
        runner.runtime.mtp_result = (
            np.array([101, 102, -1, -1, 201, 202, 203, -1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
            np.array([3, 2], dtype=np.int32),
        )

        result = runner.run(make_plan(mode="decode"))

        self.assertEqual(result.token_ids, [[101, 102], [201, 202, 203]])
        self.assertEqual(result.draft_token_counts, [3, 2])
        self.assertEqual(runner.mtp_stats(), {
            "drafted_tokens": 5,
            "accepted_tokens": 3,
            "verification_steps": 2,
        })
        self.assertEqual(len(runner.runtime.mtp_calls), 1)
        _, capacity = runner.runtime.mtp_calls[0]
        self.assertEqual(capacity, 4)
        self.assertEqual(runner.runtime.run_calls, [])

    def test_mtp_falls_back_to_one_target_step_near_context_boundary(self):
        runner = self.make_runner(
            make_config(enable_mtp=True, max_model_len=5)
        )
        runner.runtime.greedy_token_ids = np.array([777], dtype=np.int32)

        result = runner.run(make_plan(mode="decode", seq_ids=(42,)))

        self.assertEqual(result.token_ids, [[777]])
        self.assertIsNone(result.draft_token_counts)
        self.assertEqual(len(runner.runtime.run_calls), 1)
        self.assertEqual(runner.runtime.mtp_calls, [])
        self.assertEqual(runner.mtp_stats(), {
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "verification_steps": 0,
        })

    def test_sequence_slots_are_bounded_released_and_reused(self):
        runner = self.make_runner()
        runner.run(make_plan())

        runner.release_blocks([3, 1], [42])
        release = runner.runtime.release_calls[0]
        np.testing.assert_array_equal(release["block_ids"], [3, 1])
        np.testing.assert_array_equal(release["seq_ids"], [0])
        self.assertEqual(release["block_size"], 256)

        runner.run(make_plan(mode="decode", seq_ids=(77,)))
        np.testing.assert_array_equal(runner.runtime.run_calls[-1]["seq_ids"], [0])
        with self.assertRaisesRegex(RuntimeError, "slots are exhausted"):
            runner.run(make_plan(mode="decode", seq_ids=(88,)))
        with self.assertRaisesRegex(RuntimeError, "not allocated"):
            runner.release_blocks([7], [1234])

    def test_tokenizer_helpers_call_and_exit_keep_the_runner_contract(self):
        runner = self.make_runner()
        runtime = runner.runtime

        self.assertEqual(runner.call("eog_token_ids"), (248044, 248046))
        self.assertEqual(runner.call("tokenize", "hello"), [1, 5, 2])
        self.assertEqual(runner.call("detokenize", [4, 5]), "4:5")
        self.assertIsNone(runner.allocate_kv_cache(8, 256))
        runner.call("exit")
        runner.exit()
        self.assertEqual(runtime.shutdown_calls, 1)

    def test_missing_optional_tokenizer_has_a_clear_fallback(self):
        class ExecutionOnlyRuntime(FakeQwen35Runtime):
            tokenize = None

        runner = self.make_runner(runtime_type=ExecutionOnlyRuntime)
        with self.assertRaisesRegex(RuntimeError, "tokenizer_backend='hf'"):
            runner.tokenize("hello")


if __name__ == "__main__":
    unittest.main()
