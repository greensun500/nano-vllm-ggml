import unittest
from types import SimpleNamespace

import numpy as np

from nanovllm.backends.llamacpp.runner import LlamaCppRunner
from nanovllm.backends.base import BackendExecutionResult, build_execution_plan
from nanovllm.cli.chat import format_chat_prompt
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class Qwen35Tests(unittest.TestCase):

    def setUp(self):
        Sequence.block_size = 256

    def test_llamacpp_sampling_is_greedy_only(self):
        logits = np.array([[0.1, 1.2, 0.5], [3.0, 2.0, 1.0]], dtype=np.float32)
        result = LlamaCppRunner._sample(logits, np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(result, np.array([1, 0], dtype=np.int32))
        with self.assertRaises(ValueError):
            LlamaCppRunner._sample(logits, np.array([0.0, 0.1], dtype=np.float32))

    def test_llamacpp_mode_does_not_reuse_prefix_blocks(self):
        manager = BlockManager(2, 256, enable_prefix_cache=False)
        first = Sequence([7] * 256, SamplingParams(max_tokens=1))
        self.assertEqual(manager.can_allocate(first), 0)
        manager.allocate(first, 0)
        first.num_scheduled_tokens = 256
        manager.hash_blocks(first)
        manager.deallocate(first)

        second = Sequence([7] * 256, SamplingParams(max_tokens=1))
        self.assertEqual(manager.can_allocate(second), 0)
        self.assertEqual(manager.hash_to_block_id, {})

    def test_scheduler_stops_on_any_eog_token(self):
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos_token_ids=(8, 9),
            kvcache_block_size=256,
            num_kvcache_blocks=1,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=False,
            mtp_max_draft_tokens=3,
        )
        scheduler = Scheduler(config)
        sequence = Sequence([1], SamplingParams(max_tokens=4))
        scheduler.add(sequence)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, BackendExecutionResult([[9]]), is_prefill)
        self.assertTrue(sequence.is_finished)
        self.assertEqual(scheduler.pop_block_releases(), [([0], sequence.seq_id)])

    def test_qwen35_prompt_disables_thinking(self):
        prompt = format_chat_prompt(
            [{"role": "user", "content": "Hello"}],
            "Be concise.",
            "qwen35",
        )
        self.assertEqual(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_paged_kv_plan_supports_wrapped_blocks(self):
        manager = BlockManager(4, 256, enable_prefix_cache=False)

        first = Sequence([1] * 257, SamplingParams(max_tokens=1))
        manager.allocate(first, manager.can_allocate(first))
        self.assertEqual(manager.deallocate(first), [0, 1])

        second = Sequence([2], SamplingParams(max_tokens=1))
        manager.allocate(second, manager.can_allocate(second))
        self.assertEqual(manager.deallocate(second), [2])

        wrapped = Sequence([3] * 257, SamplingParams(max_tokens=1))
        manager.allocate(wrapped, manager.can_allocate(wrapped))
        self.assertEqual(wrapped.block_table, [3, 1])
        wrapped.num_scheduled_tokens = 257
        plan = build_execution_plan([wrapped], True, 256)
        self.assertEqual(plan.slot_mapping[:257], list(range(768, 1024)) + [256])

    def test_preemption_is_rejected_for_llamacpp_paged_kv(self):
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=1,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=False,
            mtp_max_draft_tokens=3,
        )
        scheduler = Scheduler(config)
        sequence = Sequence([1] * 256, SamplingParams(max_tokens=2))
        scheduler.add(sequence)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, BackendExecutionResult([[2]]), is_prefill)
        with self.assertRaisesRegex(RuntimeError, "preemption is disabled"):
            scheduler.schedule()

    def test_mtp_reserves_pages_and_commits_multiple_tokens(self):
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=2,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=True,
            mtp_max_draft_tokens=3,
        )
        scheduler = Scheduler(config)
        sequence = Sequence([1] * 253, SamplingParams(max_tokens=8))
        scheduler.add(sequence)

        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, BackendExecutionResult([[2]]), is_prefill)
        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(len(sequence.block_table), 2)

        scheduler.postprocess(
            scheduled,
            BackendExecutionResult([[3, 4, 5, 6]], [3]),
            is_prefill,
        )
        self.assertEqual(sequence.completion_token_ids, [2, 3, 4, 5, 6])
        self.assertEqual(sequence.num_cached_tokens, 257)


if __name__ == "__main__":
    unittest.main()
