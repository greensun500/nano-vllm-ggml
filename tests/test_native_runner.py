import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from nanovllm.backends.base import BackendExecutionPlan
from nanovllm.backends.native.runner import NativeRunner


class FakeQwen35Runtime:
    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.run_calls = []
        self.mtp_calls = []
        self.release_calls = []
        self.shutdown_calls = 0

    def run(self, plan):
        self.run_calls.append(plan)
        return np.array([100], dtype=np.int32)

    def run_mtp(self, plan, capacity):
        self.mtp_calls.append((plan, capacity))
        return (
            np.array([101, 102, -1, -1], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([3], dtype=np.int32),
        )

    def release_blocks(self, **kwargs):
        self.release_calls.append(kwargs)

    def eog_token_ids(self):
        return [248044, 248046]

    def execution_profile_stats(self):
        return {
            "backend": "vulkan",
            "device_name": "Mali-G720",
            "device_description": "test device",
            "profile_name": "mali-g720-vulkan",
        }

    def tokenize(self, text):
        return [1, len(text), 2]

    def detokenize(self, token_ids):
        return ":".join(str(token) for token in token_ids)

    def vocabulary_size(self):
        return 248320

    def shutdown(self):
        self.shutdown_calls += 1


def make_config(**overrides):
    values = {
        "model": "/models/Qwen3.5-2B-Q4_0.gguf",
        "gguf_model": "/models/Qwen3.5-2B-Q4_0.gguf",
        "backend": "native_cpu",
        "device_config": {"n_threads": 3, "device_index": 1},
        "max_model_len": 2048,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
        "kvcache_block_size": 256,
        "num_kvcache_blocks": 8,
        "enable_mtp": False,
        "mtp_max_draft_tokens": 3,
        "mali_experimental_graph_reuse": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_plan(mode="prefill"):
    return BackendExecutionPlan(
        mode=mode,
        input_ids=[10, 11] if mode == "prefill" else [30],
        positions=[0, 1] if mode == "prefill" else [2],
        seq_ids=[42],
        scheduled_token_counts=[2] if mode == "prefill" else [1],
        block_tables=[[3]],
        slot_mapping=[768, 769] if mode == "prefill" else [770],
        context_lens=[2] if mode == "prefill" else [3],
        num_cached_tokens=[0] if mode == "prefill" else [2],
        temperatures=[0.0],
    )


class NativeRunnerTests(unittest.TestCase):
    def make_runner(self, config=None):
        extension = ModuleType("nanovllm._C")
        extension.Qwen35Runtime = FakeQwen35Runtime
        with patch.dict(sys.modules, {"nanovllm._C": extension}):
            return NativeRunner(config or make_config())

    def test_constructor_exposes_only_release_runtime_contract(self):
        runner = self.make_runner(make_config(backend="native_vulkan"))
        self.assertEqual(runner.runtime.constructor_kwargs, {
            "model_path": "/models/Qwen3.5-2B-Q4_0.gguf",
            "backend": "vulkan",
            "max_model_len": 2048,
            "max_num_batched_tokens": 256,
            "max_num_seqs": 1,
            "block_size": 256,
            "num_blocks": 8,
            "n_threads": 3,
            "device_index": 1,
            "enable_mtp": False,
            "mtp_max_draft_tokens": 3,
            "enable_graph_reuse": True,
            "enable_vulkan_graph_reuse": False,
        })

    def test_mali_reuse_is_the_only_public_experiment(self):
        runner = self.make_runner(make_config(
            backend="native_vulkan", mali_experimental_graph_reuse=True
        ))
        self.assertTrue(runner.runtime.constructor_kwargs["enable_vulkan_graph_reuse"])

    def test_single_sequence_plan_is_converted_to_native_arrays(self):
        runner = self.make_runner()
        result = runner.run(make_plan())
        self.assertEqual(result.token_ids, [[100]])
        plan = runner.runtime.run_calls[0]
        self.assertEqual(plan["n_seqs"], 1)
        np.testing.assert_array_equal(plan["tokens"], [10, 11])
        np.testing.assert_array_equal(plan["seq_ids"], [0])
        self.assertTrue(plan["tokens"].flags.c_contiguous)

    def test_mtp_updates_stats_and_returns_accepted_tokens(self):
        runner = self.make_runner(make_config(enable_mtp=True))
        result = runner.run(make_plan("decode"))
        self.assertEqual(result.token_ids, [[101, 102]])
        self.assertEqual(result.draft_token_counts, [3])
        self.assertEqual(runner.mtp_stats(), {
            "drafted_tokens": 3,
            "accepted_tokens": 1,
            "verification_steps": 1,
        })

    def test_tokenizer_and_release_helpers_remain_native(self):
        runner = self.make_runner()
        self.assertEqual(runner.tokenize("hello"), [1, 5, 2])
        self.assertEqual(runner.detokenize([4, 5]), "4:5")
        self.assertEqual(runner.vocab_size, 248320)
        runner.run(make_plan())
        runner.release_blocks([3], [42])
        np.testing.assert_array_equal(runner.runtime.release_calls[0]["seq_ids"], [0])
        runtime = runner.runtime
        runner.exit()
        self.assertEqual(runtime.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
