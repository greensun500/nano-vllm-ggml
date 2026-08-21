import os
import unittest


class TestNativeRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from nanovllm import _C
        except ImportError as exc:
            raise unittest.SkipTest(f"native extension is not built: {exc}")
        cls.native = _C

    def test_build_has_no_llama_context(self):
        info = self.native.build_info()
        self.assertEqual(info["runtime"], "nanovllm_native")
        self.assertEqual(info["abi_version"], 1)
        self.assertEqual(
            info["ggml_commit"], "8e29a9e44f40797b2173b179949780cc98ee7176"
        )
        self.assertEqual(
            info["ggml_base_commit"], "91c631b21d6e5d09e9c6659efdf6baeef5a44ddb"
        )
        self.assertTrue(info["cpu"])
        self.assertFalse(info["cuda_graphs"])
        self.assertTrue(info["persistent_cpu_threadpool"])
        self.assertFalse(info["uses_llama_context"])

    def test_cpu_matmul_graph(self):
        result = self.native.matmul_smoke(backend="cpu", threads=2)
        self.assertEqual(result["output"], [1.0, 5.0, 2.0, 6.0, 3.0, 7.0])
        self.assertLessEqual(result["max_abs_error"], 1e-6)

    def test_cpu_q4_0_matmul_graph(self):
        result = self.native.q4_0_matmul_smoke(backend="cpu", threads=2)
        self.assertEqual(result["weight_type"], "Q4_0")
        self.assertEqual(len(result["output"]), 6)
        # Q4_0 dot products accumulate through the backend's F32 path; a small
        # reduction-order difference from the dequantized host reference is expected.
        self.assertLessEqual(result["max_abs_error"], 2e-3)

    def test_qwen35_bundled_mtp_contract(self):
        model = os.environ.get("NANOVLLM_TEST_QWEN35_GGUF")
        if not model:
            self.skipTest("NANOVLLM_TEST_QWEN35_GGUF is not set")
        result = self.native.qwen35_model_info(path=model)
        self.assertEqual(result["architecture"], "qwen35")
        self.assertEqual(result["block_count"], 25)
        self.assertEqual(result["main_layers"], 24)
        self.assertEqual(result["nextn_predict_layers"], 1)
        self.assertEqual(result["tensor_count"], 335)
        self.assertTrue(result["strict_q4_0_profile"])
        self.assertEqual(result["full_attention_layers"], [3, 7, 11, 15, 19, 23])
        self.assertEqual(len(result["recurrent_layers"]), 18)


if __name__ == "__main__":
    unittest.main()
