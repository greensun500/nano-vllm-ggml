import unittest
from types import SimpleNamespace

import numpy as np

from nanovllm.backends.llamacpp.runner import LlamaCppRunner
from nanovllm.backends.base import BackendExecutionResult, build_execution_plan
from nanovllm.cli.chat import format_chat_prompt
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
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
            max_model_len=256,
        )
        scheduler = Scheduler(config)
        sequence = Sequence([1], SamplingParams(max_tokens=4))
        scheduler.add(sequence)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, BackendExecutionResult([[9]]), is_prefill)
        self.assertTrue(sequence.is_finished)
        self.assertEqual(scheduler.pop_block_releases(), [([0], sequence.seq_id)])

    def test_native_vulkan_prefill_keeps_one_sequence_in_one_plan(self):
        config = SimpleNamespace(
            backend="native_vulkan", max_num_seqs=1, max_num_batched_tokens=64,
            eos_token_ids=(), kvcache_block_size=256, num_kvcache_blocks=3,
            enable_prefix_cache=False, enable_preemption=False, enable_mtp=False,
            mtp_max_draft_tokens=3, max_model_len=768,
        )
        scheduler = Scheduler(config)
        sequence = Sequence([1] * 521, SamplingParams(max_tokens=1))
        scheduler.add(sequence)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [sequence])
        self.assertEqual(sequence.num_scheduled_tokens, 521)

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
            max_model_len=512,
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
            max_model_len=512,
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

    def test_session_sequence_parks_then_prefills_only_its_pending_token_and_new_turn(self):
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
            max_model_len=256,
            enable_session_cache=True,
            max_retained_sessions=1,
            max_consecutive_prefill_rounds=0,
        )
        scheduler = Scheduler(config)
        sequence = Sequence([1], SamplingParams(max_tokens=1))
        sequence.retain_cache = True
        scheduler.add(sequence)

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        scheduler.postprocess(scheduled, BackendExecutionResult([[2]]), is_prefill)
        self.assertEqual(sequence.status, SequenceStatus.PARKED)
        self.assertEqual(sequence.num_cached_tokens, 1)
        self.assertEqual(sequence.token_ids, [1, 2])
        self.assertEqual(scheduler.pop_block_releases(), [])

        sequence.begin_turn([3, 4], SamplingParams(max_tokens=1))
        scheduler.resume(sequence)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(sequence.num_scheduled_tokens, 3)
        plan = build_execution_plan(scheduled, is_prefill, 256)
        self.assertEqual(plan.input_ids, [2, 3, 4])
        scheduler.postprocess(scheduled, BackendExecutionResult([[5]]), is_prefill)
        self.assertEqual(sequence.status, SequenceStatus.PARKED)
        self.assertEqual(sequence.turn_completion_token_ids, [5])

        scheduler.close_parked(sequence)
        self.assertEqual(sequence.status, SequenceStatus.FINISHED)
        self.assertEqual(scheduler.pop_block_releases(), [([0], sequence.seq_id)])

    def test_session_cache_evicts_the_oldest_parked_sequence(self):
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=2,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=False,
            mtp_max_draft_tokens=3,
            max_model_len=256,
            enable_session_cache=True,
            max_retained_sessions=1,
            max_consecutive_prefill_rounds=0,
        )
        scheduler = Scheduler(config)
        first = Sequence([1], SamplingParams(max_tokens=1))
        second = Sequence([2], SamplingParams(max_tokens=1))
        first.retain_cache = second.retain_cache = True

        for sequence, output in ((first, 11), (second, 12)):
            scheduler.add(sequence)
            scheduled, is_prefill = scheduler.schedule()
            scheduler.postprocess(scheduled, BackendExecutionResult([[output]]), is_prefill)

        self.assertEqual(first.status, SequenceStatus.FINISHED)
        self.assertEqual(second.status, SequenceStatus.PARKED)
        self.assertEqual(scheduler.pop_block_releases(), [([0], first.seq_id)])

    def test_resumed_session_expands_its_block_table_before_prefill(self):
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=512,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=2,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=False,
            mtp_max_draft_tokens=3,
            max_model_len=512,
            enable_session_cache=True,
            max_retained_sessions=1,
            max_consecutive_prefill_rounds=0,
        )
        scheduler = Scheduler(config)
        sequence = Sequence(list(range(256)), SamplingParams(max_tokens=1))
        sequence.retain_cache = True
        scheduler.add(sequence)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, BackendExecutionResult([[256]]), is_prefill)
        self.assertEqual(sequence.status, SequenceStatus.PARKED)
        self.assertEqual(sequence.num_cached_tokens, 256)
        self.assertEqual(sequence.block_table, [0])

        sequence.begin_turn([257], SamplingParams(max_tokens=1))
        scheduler.resume(sequence)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(sequence.block_table, [0, 1])
        self.assertEqual(sequence.num_scheduled_tokens, 2)
        self.assertEqual(build_execution_plan(scheduled, is_prefill, 256).input_ids, [256, 257])

    def test_new_native_request_evicts_idle_session_when_its_slot_is_needed(self):
        config = SimpleNamespace(
            backend="native_cpu",
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=2,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=False,
            mtp_max_draft_tokens=3,
            max_model_len=256,
            enable_session_cache=True,
            max_retained_sessions=1,
            max_consecutive_prefill_rounds=0,
        )
        scheduler = Scheduler(config)
        parked = Sequence([1], SamplingParams(max_tokens=1))
        parked.retain_cache = True
        scheduler.add(parked)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, BackendExecutionResult([[2]]), is_prefill)
        self.assertEqual(parked.status, SequenceStatus.PARKED)

        incoming = Sequence([3], SamplingParams(max_tokens=1))
        scheduler.add(incoming)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [incoming])
        self.assertEqual(parked.status, SequenceStatus.FINISHED)
        self.assertEqual(scheduler.pop_block_releases(), [([0], parked.seq_id)])

    def test_new_request_evicts_idle_session_when_its_blocks_are_needed(self):
        config = SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=256,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=1,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=False,
            mtp_max_draft_tokens=3,
            max_model_len=256,
            enable_session_cache=True,
            max_retained_sessions=1,
            max_consecutive_prefill_rounds=0,
        )
        scheduler = Scheduler(config)
        parked = Sequence([1], SamplingParams(max_tokens=1))
        parked.retain_cache = True
        scheduler.add(parked)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, BackendExecutionResult([[2]]), is_prefill)

        incoming = Sequence([3], SamplingParams(max_tokens=1))
        scheduler.add(incoming)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [incoming])
        self.assertEqual(parked.status, SequenceStatus.FINISHED)
        self.assertEqual(scheduler.pop_block_releases(), [([0], parked.seq_id)])

    def test_decode_runs_after_the_configured_prefill_budget(self):
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=2,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=False,
            mtp_max_draft_tokens=3,
            max_model_len=256,
            enable_session_cache=False,
            max_retained_sessions=0,
            max_consecutive_prefill_rounds=1,
        )
        scheduler = Scheduler(config)
        running = Sequence([1], SamplingParams(max_tokens=4))
        scheduler.add(running)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, BackendExecutionResult([[2]]), is_prefill)

        waiting = Sequence([3, 4], SamplingParams(max_tokens=4))
        scheduler.add(waiting)
        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [waiting])
        scheduler.postprocess(scheduled, BackendExecutionResult([[5]]), is_prefill)

        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [running])


if __name__ == "__main__":
    unittest.main()
