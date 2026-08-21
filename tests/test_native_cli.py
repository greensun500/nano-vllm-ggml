import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.backends.base import BackendExecutionResult
from nanovllm.cli import bench, chat
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.llm_engine import _hf_eog_token_ids
from nanovllm.sampling_params import SamplingParams


def make_bench_args(**overrides):
    values = {
        "backend": "native_cpu",
        "batch_size": 1,
        "gen_len": 32,
        "repeat": 1,
        "warmup": 0,
        "max_num_seqs": 1,
        "prompt_len": 128,
        "max_model_len": 2048,
        "temperature": 0.0,
        "use_prefix_cache": False,
        "enable_mtp": False,
        "mtp_max_draft_tokens": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeMtpRunner:
    def __init__(self):
        self.plans = []

    def call(self, method_name, plan):
        if method_name != "run":
            raise AssertionError(f"unexpected runner method: {method_name}")
        self.plans.append(plan)
        if plan.is_prefill:
            return BackendExecutionResult([[101]])
        # The request has room for only one more completion token. Scheduler
        # must retain 102 and discard the remaining verified MTP output.
        return BackendExecutionResult([[102, 103, 104, 105]], [3])


class FakeLlm:
    def __init__(self):
        self.config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=1,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=True,
            mtp_max_draft_tokens=3,
            max_model_len=256,
        )
        self.scheduler = Scheduler(self.config)
        self.model_runner = FakeMtpRunner()

    def add_request(self, prompt, sampling_params):
        self.scheduler.add(Sequence(prompt, sampling_params))

    def is_finished(self):
        return self.scheduler.is_finished()

    def flush_backend_releases(self):
        self.scheduler.pop_block_releases()


class NativeBenchCliTests(unittest.TestCase):
    def test_bench_parser_accepts_mali_mtp_prefill_strategy(self):
        with patch.object(
            sys,
            "argv",
            ["bench.py", "--native-mali-mtp-prefill-strategy", "chunked_staged"],
        ):
            args = bench.parse_args()
        self.assertEqual(args.native_mali_mtp_prefill_strategy, "chunked_staged")

    def assert_rejected_before_build(self, args, message):
        with patch.object(bench, "build_llm") as build_llm:
            with self.assertRaisesRegex(SystemExit, message):
                bench.benchmark(args)
            build_llm.assert_not_called()

    def test_non_cuda_temperature_is_rejected_before_model_load(self):
        for backend in (
            "native_cpu",
            "native_vulkan",
            "llamacpp_cpu",
            "llamacpp_vulkan",
        ):
            with self.subTest(backend=backend):
                self.assert_rejected_before_build(
                    make_bench_args(backend=backend, temperature=0.1),
                    "require --temperature 0",
                )

    def test_non_cuda_prefix_cache_is_rejected_before_model_load(self):
        for backend in (
            "native_cpu",
            "native_vulkan",
            "llamacpp_cpu",
            "llamacpp_vulkan",
        ):
            with self.subTest(backend=backend):
                self.assert_rejected_before_build(
                    make_bench_args(backend=backend, use_prefix_cache=True),
                    "supported only by the CUDA backend",
                )

    def test_context_overflow_is_rejected_before_model_load(self):
        self.assert_rejected_before_build(
            make_bench_args(prompt_len=128, gen_len=129, max_model_len=256),
            r"--prompt-len \+ --gen-len must be <= --max-model-len",
        )

    def test_decode_tokens_count_only_scheduler_retained_mtp_output(self):
        Sequence.block_size = 256
        llm = FakeLlm()

        result = bench.run_generation_once(
            llm,
            prompts=[[7]],
            sampling_params=[
                SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=2)
            ],
        )

        self.assertEqual(len(llm.model_runner.plans), 2)
        self.assertTrue(llm.model_runner.plans[0].is_prefill)
        self.assertFalse(llm.model_runner.plans[1].is_prefill)
        self.assertEqual(result["prefill_tokens"], 1)
        self.assertEqual(result["generated_tokens"], 2)
        self.assertEqual(result["decode_tokens"], 1)

    def test_scheduler_does_not_reserve_full_mtp_window_past_context_boundary(self):
        Sequence.block_size = 256
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos_token_ids=(),
            kvcache_block_size=256,
            num_kvcache_blocks=1,
            enable_prefix_cache=False,
            enable_preemption=False,
            enable_mtp=True,
            mtp_max_draft_tokens=3,
            max_model_len=256,
        )
        scheduler = Scheduler(config)
        sequence = Sequence(
            [7] * 254,
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=2),
        )
        scheduler.add(sequence)
        prefill, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        scheduler.postprocess(prefill, BackendExecutionResult([[8]]), True)

        decode, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(decode, [sequence])
        self.assertEqual(len(sequence.block_table), 1)


class NativeChatCliTests(unittest.TestCase):
    def test_chat_parser_accepts_mali_mtp_prefill_strategy(self):
        with patch.object(
            sys,
            "argv",
            ["chat.py", "--native-mali-mtp-prefill-strategy", "chunked_legacy"],
        ):
            args = chat.parse_args()
        self.assertEqual(args.native_mali_mtp_prefill_strategy, "chunked_legacy")

    @staticmethod
    def make_args(**overrides):
        values = {
            "backend": "native_cpu",
            "temperature": 0.0,
            "max_tokens": 32,
            "enable_mtp": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_invalid_chat_sampling_is_rejected_before_model_load(self):
        for args, message in (
            (self.make_args(max_tokens=0), "--max-tokens must be positive"),
            (self.make_args(temperature=0.1), "require --temperature 0"),
            (
                self.make_args(backend="cuda", enable_mtp=True),
                "supported only by CPU/Vulkan GGUF backends",
            ),
        ):
            with self.subTest(args=args), patch.object(chat, "build_llm") as build_llm:
                with self.assertRaisesRegex(SystemExit, message):
                    chat.validate_args(args)
                build_llm.assert_not_called()

    def test_sampling_params_reject_non_positive_max_tokens(self):
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "max_tokens must be positive"
            ):
                SamplingParams(max_tokens=value)

    def test_native_qwen_hf_eog_contract_includes_both_text_terminators(self):
        tokenizer = SimpleNamespace(
            eos_token_id=248046,
            unk_token_id=None,
            convert_tokens_to_ids=lambda token: {
                "<|endoftext|>": 248044,
                "<|im_end|>": 248046,
            }[token],
        )
        self.assertEqual(
            _hf_eog_token_ids(tokenizer, include_qwen_eog=True),
            (248046, 248044),
        )


if __name__ == "__main__":
    unittest.main()
