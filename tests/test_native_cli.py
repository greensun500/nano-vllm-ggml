import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.cli import chat


class NativeCliTests(unittest.TestCase):
    def test_chat_turn_metrics_are_compact_and_handle_no_decode_step(self):
        self.assertEqual(
            chat.format_turn_metrics({
                "prefill_tokens": 120,
                "prefill_seconds": 0.5,
                "prefill_generated_tokens": 1,
                "decode_tokens": 0,
                "decode_seconds": 0.0,
                "generated_tokens": 1,
            }),
            "[timing] prefill: 240.00 tok/s (120 tok, 500.0 ms) | "
            "decode: n/a | generated: 1 tok (prefill 1 + decode 0)",
        )

    def test_chat_parser_exposes_only_native_runtime_controls(self):
        with patch.object(
            sys,
            "argv",
            [
                "chat.py",
                "model.gguf",
                "--backend",
                "native_vulkan",
                "--enable-mtp",
                "--mtp-max-draft-tokens",
                "3",
                "--mali-experimental-graph-reuse",
            ],
        ):
            args = chat.parse_args()
        self.assertEqual(args.backend, "native_vulkan")
        self.assertTrue(args.enable_mtp)
        self.assertEqual(args.mtp_max_draft_tokens, 3)
        self.assertTrue(args.mali_experimental_graph_reuse)

    def test_chat_build_llm_uses_gguf_only(self):
        args = SimpleNamespace(
            model="model.gguf",
            gguf_model=None,
            backend="native_cpu",
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=1,
            num_kvcache_blocks=2,
            enable_mtp=False,
            mtp_max_draft_tokens=1,
            mali_experimental_graph_reuse=False,
            threads=2,
            device_index=0,
        )
        with patch("nanovllm.cli.chat.LLM", return_value=object()) as llm:
            chat.build_llm(args)
        self.assertEqual(
            llm.call_args.kwargs,
            {
                "backend": "native_cpu",
                "gguf_model": "model.gguf",
                "max_model_len": 512,
                "max_num_batched_tokens": 512,
                "max_num_seqs": 1,
                "num_kvcache_blocks": 2,
                "enable_mtp": False,
                "mtp_max_draft_tokens": 1,
                "mali_experimental_graph_reuse": False,
                "device_config": {"n_threads": 2, "device_index": 0},
            },
        )


if __name__ == "__main__":
    unittest.main()
