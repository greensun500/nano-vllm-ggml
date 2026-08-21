"""Opt-in real-GGUF tokenization compatibility gate for the native runtime."""

from __future__ import annotations

import os
import unittest
from pathlib import Path


MODEL_ENV = "NANOVLLM_TEST_QWEN35_GGUF"
TOKENIZER_ENV = "NANOVLLM_TEST_QWEN35_TOKENIZER"


class TestNativeQwen35GgufTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_value = os.environ.get(MODEL_ENV)
        if not model_value:
            raise unittest.SkipTest(f"set {MODEL_ENV} to run this real-GGUF gate")
        cls.model_path = Path(model_value)
        if not cls.model_path.is_file():
            raise unittest.SkipTest(f"GGUF does not exist: {cls.model_path}")

        try:
            from nanovllm import _C
        except ImportError as exc:
            raise unittest.SkipTest(f"native extension is not built: {exc}") from exc
        if not hasattr(_C, "Qwen35Runtime"):
            raise unittest.SkipTest("native extension does not expose Qwen35Runtime")

        cls.runtime = _C.Qwen35Runtime(
            model_path=os.fspath(cls.model_path),
            backend="cpu",
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=1,
            block_size=256,
            num_blocks=2,
            n_threads=max(1, int(os.environ.get("NANOVLLM_TEST_CPU_THREADS", "1"))),
            device_index=0,
            enable_mtp=False,
            mtp_max_draft_tokens=3,
            attention_impl="math",
        )

    @classmethod
    def tearDownClass(cls):
        runtime = getattr(cls, "runtime", None)
        if runtime is not None:
            runtime.shutdown()

    def test_embedded_tokenizer_round_trips_text_and_reports_eog(self):
        for text in (
            "Hello, 世界!",
            "<|im_start|>user\n你好，world!<|im_end|>\n",
            "  emoji: 😀\n",
        ):
            with self.subTest(text=text):
                token_ids = self.runtime.tokenize(text)
                self.assertTrue(token_ids)
                self.assertEqual(self.runtime.detokenize(token_ids), text)
        self.assertEqual(self.runtime.vocabulary_size(), 248320)
        self.assertTrue({248044, 248046}.issubset(self.runtime.eog_token_ids()))

    def test_embedded_tokenizer_matches_hf_when_an_oracle_is_available(self):
        tokenizer_path = os.environ.get(TOKENIZER_ENV)
        if not tokenizer_path or not Path(tokenizer_path).is_dir():
            self.skipTest(f"set {TOKENIZER_ENV} to compare against Hugging Face")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            self.skipTest(f"transformers is unavailable: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        for text in (
            "Hello, 世界!",
            "<|im_start|>user\n你好，world!<|im_end|>\n",
            "  emoji: 😀\n",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    self.runtime.tokenize(text),
                    tokenizer.encode(text, add_special_tokens=True),
                )


if __name__ == "__main__":
    unittest.main()
