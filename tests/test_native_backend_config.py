import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.backends import create_backend
from nanovllm.config import Config


class NativeBackendConfigTests(unittest.TestCase):

    def make_model(self):
        return tempfile.NamedTemporaryFile(suffix=".gguf")

    def test_native_cpu_derives_gguf_and_disables_unsupported_scheduler_features(self):
        with self.make_model() as model, tempfile.TemporaryDirectory() as tokenizer:
            config = Config(
                model=model.name,
                backend="native_cpu",
                tokenizer=tokenizer,
                max_model_len=513,
            )

        self.assertEqual(config.gguf_model, model.name)
        self.assertEqual(config.model_format, "gguf")
        self.assertFalse(config.enable_prefix_cache)
        self.assertFalse(config.enable_preemption)
        self.assertEqual(config.tokenizer, tokenizer)
        self.assertEqual(config.tokenizer_backend, "hf")
        self.assertEqual(config.num_kvcache_blocks, 3)
        self.assertEqual(config.max_model_len, 513)
        self.assertEqual(config.max_num_seqs, 1)
        self.assertEqual(config.max_num_batched_tokens, 2048)

    def test_native_backend_requires_the_explicit_hf_tokenizer_contract(self):
        with self.make_model() as model, tempfile.TemporaryDirectory() as tokenizer:
            with self.assertRaisesRegex(ValueError, "tokenizer_backend='hf'"):
                Config(
                    model=model.name,
                    backend="native_vulkan",
                    tokenizer=tokenizer,
                    tokenizer_backend="llamacpp",
                )
            with self.assertRaisesRegex(ValueError, "requires tokenizer="):
                Config(model=model.name, backend="native_vulkan")
            with self.assertRaisesRegex(ValueError, "local Hugging Face tokenizer directory"):
                Config(
                    model=model.name,
                    backend="native_vulkan",
                    tokenizer="/does/not/exist",
                )

    def test_native_backend_rejects_prefix_cache_and_preemption(self):
        with self.make_model() as model:
            with self.assertRaisesRegex(ValueError, "native backend does not support prefix caching"):
                Config(model=model.name, backend="native_cpu", enable_prefix_cache=True)
            with self.assertRaisesRegex(ValueError, "native backend does not support preemption"):
                Config(model=model.name, backend="native_cpu", enable_preemption=True)

    def test_native_mtp_keeps_the_existing_verification_batch_contract(self):
        with self.make_model() as model, tempfile.TemporaryDirectory() as tokenizer:
            with self.assertRaisesRegex(ValueError, "must fit one MTP verification batch"):
                Config(
                    model=model.name,
                    backend="native_cpu",
                    tokenizer=tokenizer,
                    enable_mtp=True,
                    mtp_max_draft_tokens=3,
                    max_num_seqs=2,
                    max_num_batched_tokens=7,
                )

    def test_native_execution_optimization_options_are_explicit_and_validated(self):
        with self.make_model() as model, tempfile.TemporaryDirectory() as tokenizer:
            config = Config(
                model=model.name,
                backend="native_cpu",
                tokenizer=tokenizer,
                enable_mtp=True,
                native_attention_impl="auto",
                native_batched_recurrent_snapshots=True,
                native_mtp_prefill_fusion=True,
            )
            self.assertEqual(config.native_attention_impl, "auto")
            self.assertTrue(config.native_batched_recurrent_snapshots)
            self.assertTrue(config.native_mtp_prefill_fusion)

            with self.assertRaisesRegex(ValueError, "native_vulkan_graph_reuse"):
                Config(
                    model=model.name,
                    backend="native_cpu",
                    tokenizer=tokenizer,
                    native_vulkan_graph_reuse=True,
                )

            with self.assertRaisesRegex(ValueError, "native_attention_impl"):
                Config(
                    model=model.name,
                    backend="native_cpu",
                    tokenizer=tokenizer,
                    native_attention_impl="invalid",
                )
            with self.assertRaisesRegex(ValueError, "requires enable_mtp=True"):
                Config(
                    model=model.name,
                    backend="native_cpu",
                    tokenizer=tokenizer,
                    native_mtp_prefill_fusion=True,
                )

    def test_explicit_physical_blocks_do_not_change_per_sequence_context(self):
        with self.make_model() as model, tempfile.TemporaryDirectory() as tokenizer:
            config = Config(
                model=model.name,
                backend="native_cpu",
                tokenizer=tokenizer,
                max_model_len=300,
                num_kvcache_blocks=8,
            )
            self.assertEqual(config.max_model_len, 300)
            self.assertEqual(config.num_kvcache_blocks, 8)

            with self.assertRaisesRegex(ValueError, "must cover max_model_len"):
                Config(
                    model=model.name,
                    backend="native_cpu",
                    tokenizer=tokenizer,
                    max_model_len=300,
                    num_kvcache_blocks=1,
                )

    def test_invalid_capacity_values_raise_explicit_errors(self):
        cases = (
            ({"max_model_len": 0}, "max_model_len must be positive"),
            ({"max_num_batched_tokens": 0}, "max_num_batched_tokens must be positive"),
            ({"max_num_seqs": 0}, "max_num_seqs must be positive"),
            ({"kvcache_block_size": 0}, "kvcache_block_size must be positive"),
            ({"num_kvcache_blocks": 0}, "num_kvcache_blocks must be positive or -1"),
            ({"num_kvcache_blocks": -2}, "num_kvcache_blocks must be positive or -1"),
            (
                {"max_num_batched_tokens": 1, "max_num_seqs": 2},
                "must be at least max_num_seqs",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, message
            ):
                Config(model="unused.gguf", backend="native_cpu", **overrides)

    def test_factory_constructs_native_runner_for_all_native_backends(self):
        for backend in ("native_cpu", "native_vulkan", "native_cuda"):
            config = SimpleNamespace(backend=backend)
            sentinel = object()
            with self.subTest(backend=backend), patch(
                "nanovllm.backends.native.runner.NativeRunner",
                return_value=sentinel,
            ) as runner:
                self.assertIs(create_backend(config), sentinel)
                runner.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
