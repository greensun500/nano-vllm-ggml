import unittest
from types import SimpleNamespace

import numpy as np

from nanovllm.backends.llamacpp.runner import LlamaCppRunner
from nanovllm.cli.chat import format_chat_prompt
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class Qwen35Stage1Tests(unittest.TestCase):

    def setUp(self):
        Sequence.block_size = 256

    def test_llamacpp_sampling_is_greedy_only(self):
        logits = np.array([[0.1, 1.2, 0.5], [3.0, 2.0, 1.0]], dtype=np.float32)
        result = LlamaCppRunner._sample(logits, np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(result, np.array([1, 0], dtype=np.int32))
        with self.assertRaises(ValueError):
            LlamaCppRunner._sample(logits, np.array([0.0, 0.1], dtype=np.float32))

    def test_native_kv_mode_does_not_reuse_prefix_blocks(self):
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
        )
        scheduler = Scheduler(config)
        sequence = Sequence([1], SamplingParams(max_tokens=4))
        scheduler.add(sequence)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [9], is_prefill)
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


if __name__ == "__main__":
    unittest.main()
