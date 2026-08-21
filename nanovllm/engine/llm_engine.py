import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm

from nanovllm.config import Config
from nanovllm.backends import create_backend
from nanovllm.backends.base import build_execution_plan
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler


def _hf_eog_token_ids(tokenizer, include_qwen_eog: bool = False) -> tuple[int, ...]:
    eos = tokenizer.eos_token_id
    values = [eos] if isinstance(eos, int) else list(eos or ())
    if include_qwen_eog:
        unknown = getattr(tokenizer, "unk_token_id", None)
        for token in ("<|endoftext|>", "<|im_end|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != unknown:
                values.append(token_id)
    return tuple(dict.fromkeys(values))


class ChatSession:
    """A retained single-user chat context backed by one native sequence slot."""

    def __init__(self, engine: "LLMEngine", sequence: "Sequence"):
        self._engine = engine
        self._sequence = sequence

    @property
    def session_id(self) -> int:
        return self._sequence.seq_id

    @property
    def is_open(self) -> bool:
        return self._sequence.status == SequenceStatus.PARKED

    def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        use_tqdm: bool = False,
    ) -> dict:
        return self._engine._continue_session(self._sequence, prompt, sampling_params, use_tqdm)

    def close(self):
        self._engine._close_session(self._sequence)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}#取出config的字段名称，写成一个集合
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}#按照字段，取出key-value对，写成一个字典
        config = Config(model, **config_kwargs)#config初始化
        Sequence.block_size = config.kvcache_block_size#设定kvcache的block——size
        self.ps = []#保存子进程对象
        self.events = []#保存进程间同步的事件
        if config.backend == "cuda":
            import torch.multiprocessing as mp

            ctx = mp.get_context("spawn")#获取ctx对象，后续创建进程都以spawn方式启动
            for i in range(1, config.tensor_parallel_size):#以tensor_parallel_size为指标，创建多个子进程。为后续的多卡推理功能做准备
                event = ctx.Event()
                process = ctx.Process(target=create_backend, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = create_backend(config, 0, self.events)#主进程会把模型初始化成功，然后创建共享内存之后返回来，子进程则在里面等待任务
        else:
            self.model_runner = create_backend(config)
        self.config = config
        self.closed = False
        self._runtime_metrics = {
            "tokenize_seconds": 0.0,
            "schedule_seconds": 0.0,
            "plan_seconds": 0.0,
            "native_run_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "steps": 0,
        }
        if config.tokenizer_backend in ("llamacpp", "native"):
            self.tokenizer = None
            config.eos_token_ids = self.model_runner.call("eog_token_ids")
        else:
            from transformers import AutoTokenizer

            tokenizer_path = config.tokenizer or config.model
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            config.eos_token_ids = _hf_eog_token_ids(
                self.tokenizer,
                include_qwen_eog=config.backend in ("native_cpu", "native_vulkan", "native_cuda"),
            )
        self.scheduler = Scheduler(config)#调度器
        atexit.register(self.exit)#整个程序退出时候字段调用

    def exit(self):
        if self.closed:
            return
        self.closed = True
        self.model_runner.call("exit")
        if hasattr(self, "model_runner"):
            del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        prompt = self._tokenize_prompt(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def start_session(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        use_tqdm: bool = False,
    ) -> tuple[ChatSession, dict]:
        if not self.config.enable_session_cache:
            raise RuntimeError("session cache is disabled; set enable_session_cache=True")
        seq = Sequence(self._tokenize_prompt(prompt), sampling_params)
        seq.retain_cache = True
        self.scheduler.add(seq)
        output = self._run_session_turn(seq, use_tqdm)
        return ChatSession(self, seq), output

    def _continue_session(
        self,
        seq: "Sequence",
        prompt: str | list[int],
        sampling_params: SamplingParams,
        use_tqdm: bool,
    ) -> dict:
        if self.closed:
            raise RuntimeError("LLM engine is closed")
        token_ids = self._tokenize_prompt(prompt)
        if len(seq) + len(token_ids) > self.config.max_model_len:
            raise ValueError("chat turn exceeds max_model_len")
        seq.begin_turn(token_ids, sampling_params)
        self.scheduler.resume(seq)
        return self._run_session_turn(seq, use_tqdm)

    def _close_session(self, seq: "Sequence"):
        if seq.status != SequenceStatus.PARKED:
            raise RuntimeError("chat session is already closed or still running")
        self.scheduler.close_parked(seq)
        self.flush_backend_releases()

    def _tokenize_prompt(self, prompt: str | list[int]) -> list[int]:
        if not isinstance(prompt, str):
            return prompt
        started = perf_counter()
        try:
            if self.config.tokenizer_backend in ("llamacpp", "native"):
                return self.model_runner.call("tokenize", prompt)
            return self.tokenizer.encode(prompt)
        finally:
            self._runtime_metrics["tokenize_seconds"] += perf_counter() - started

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if self.config.tokenizer_backend in ("llamacpp", "native"):
            return self.model_runner.call("detokenize", token_ids)
        return self.tokenizer.decode(token_ids)

    def _run_session_turn(self, seq: "Sequence", use_tqdm: bool) -> dict:
        pbar = tqdm(total=seq.max_tokens, desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        try:
            while seq.status != SequenceStatus.PARKED:
                if seq.is_finished:
                    raise RuntimeError("chat session was evicted before its turn completed")
                completed_before = seq.num_turn_completion_tokens
                self.step()
                pbar.update(seq.num_turn_completion_tokens - completed_before)
        finally:
            pbar.close()
        token_ids = list(seq.turn_completion_token_ids)
        return {"text": self._decode_tokens(token_ids), "token_ids": token_ids}

    def flush_backend_releases(self):
        for block_ids, seq_id in self.scheduler.pop_block_releases():
            self.model_runner.call("release_blocks", block_ids, [seq_id])

    def runtime_metrics(self) -> dict[str, int | float]:
        metrics = dict(self._runtime_metrics)
        memory_stats = getattr(self.model_runner, "memory_stats", None)
        if callable(memory_stats):
            metrics.update({f"native_{key}": value for key, value in memory_stats().items()})
        return metrics

    def step(self):
        started = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()#调度返回这一轮要计算的seq列表（包括prompt+采样参数），以及是否是prefill
        self._runtime_metrics["schedule_seconds"] += perf_counter() - started
        self.flush_backend_releases()
        started = perf_counter()
        plan = build_execution_plan(seqs, is_prefill, self.config.kvcache_block_size)
        self._runtime_metrics["plan_seconds"] += perf_counter() - started
        started = perf_counter()
        result = self.model_runner.call("run", plan)
        self._runtime_metrics["native_run_seconds"] += perf_counter() - started
        num_tokens = (sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill
                      else -sum(len(token_ids) for token_ids in result.token_ids))#计算这轮处理了多少token数据
        started = perf_counter()
        self.scheduler.postprocess(seqs, result, is_prefill)
        self._runtime_metrics["postprocess_seconds"] += perf_counter() - started
        self._runtime_metrics["steps"] += 1
        self.flush_backend_releases()
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(#获取prompt、#获取采样参数，开始进入主循环，开始推理
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):#如果采样参数不是列表，则把采样参数复制len(prompts)次
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)#依次入队prompt、采样参数作为调度器任务
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()#记录开始时间
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self._decode_tokens(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
