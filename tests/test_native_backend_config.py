import tempfile
import unittest

from nanovllm.config import Config


class NativeBackendConfigTests(unittest.TestCase):
    def make_model(self):
        return tempfile.NamedTemporaryFile(suffix=".gguf")

    def test_native_gguf_defaults_are_single_sequence_and_self_contained(self):
        with self.make_model() as model:
            config = Config(model=model.name, max_model_len=513)
        self.assertEqual(config.backend, "native_cpu")
        self.assertEqual(config.gguf_model, model.name)
        self.assertEqual(config.max_num_seqs, 1)
        self.assertEqual(config.num_kvcache_blocks, 3)
        self.assertFalse(config.enable_mtp)
        self.assertEqual(config.mtp_max_draft_tokens, 1)

    def test_only_native_backends_and_one_sequence_are_public(self):
        with self.make_model() as model:
            with self.assertRaisesRegex(ValueError, "unsupported native backend"):
                Config(model=model.name, backend="cuda")
            with self.assertRaisesRegex(ValueError, "max_num_seqs=1"):
                Config(model=model.name, max_num_seqs=2)

    def test_mtp_and_capacity_contract(self):
        with self.make_model() as model:
            with self.assertRaisesRegex(ValueError, "must fit one MTP"):
                Config(
                    model=model.name,
                    enable_mtp=True,
                    mtp_max_draft_tokens=3,
                    max_num_batched_tokens=3,
                )
            config = Config(
                model=model.name,
                enable_mtp=True,
                mtp_max_draft_tokens=3,
                max_num_batched_tokens=4,
            )
            self.assertTrue(config.enable_mtp)

    def test_mali_graph_reuse_is_explicit_and_vulkan_only(self):
        with self.make_model() as model:
            with self.assertRaisesRegex(ValueError, "requires backend='native_vulkan'"):
                Config(model=model.name, mali_experimental_graph_reuse=True)
            config = Config(
                model=model.name,
                backend="native_vulkan",
                mali_experimental_graph_reuse=True,
            )
            self.assertTrue(config.mali_experimental_graph_reuse)

    def test_invalid_capacity_and_scheduler_values_are_rejected(self):
        with self.make_model() as model:
            for overrides, message in (
                ({"max_model_len": 0}, "max_model_len must be positive"),
                ({"max_num_batched_tokens": 0}, "max_num_batched_tokens must be positive"),
                ({"kvcache_block_size": 0}, "kvcache_block_size must be positive"),
                ({"num_kvcache_blocks": 0}, "num_kvcache_blocks must be positive"),
                ({"enable_prefix_cache": True}, "does not support prefix caching"),
                ({"enable_preemption": True}, "does not support preemption"),
            ):
                with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                    Config(model=model.name, **overrides)


if __name__ == "__main__":
    unittest.main()
