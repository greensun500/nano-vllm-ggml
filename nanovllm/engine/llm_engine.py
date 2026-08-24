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
        unsupported = sorted(set(kwargs) - config_fields)
        if unsupported:
            raise TypeError(
                "unsupported native runtime options: " + ", ".join(unsupported)
            )
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}#按照字段，取出key-value对，写成一个字典
        config = Config(model, **config_kwargs)#config初始化
        Sequence.block_size = config.kvcache_block_size#设定kvcache的block——size
        self.ps = []#保存子进程对象
        self.events = []#保存进程间同步的事件
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
        self.tokenizer = None
        config.eos_token_ids = self.model_runner.call("eog_token_ids")
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
            return self.model_runner.call("tokenize", prompt)
        finally:
            self._runtime_metrics["tokenize_seconds"] += perf_counter() - started

    def _decode_tokens(self, token_ids: list[int]) -> str:
        return self.model_runner.call("detokenize", token_ids)

    def _run_session_turn(self, seq: "Sequence", use_tqdm: bool) -> dict:
        # Keep chat timing scoped to this turn.  Measure the same execution
        # envelope as the benchmark command (schedule, plan, native run and
        # postprocess), while excluding tokenization, detokenization and CLI
        # output.  ``step`` reports prompt tokens as positive and generated
        # decode tokens as negative.
        turn_metrics = {
            "prefill_tokens": 0,
            "prefill_seconds": 0.0,
            "prefill_generated_tokens": 0,
            "decode_tokens": 0,
            "decode_seconds": 0.0,
        }
        pbar = tqdm(total=seq.max_tokens, desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        try:
            while seq.status != SequenceStatus.PARKED:
                if seq.is_finished:
                    raise RuntimeError("chat session was evicted before its turn completed")
                completed_before = seq.num_turn_completion_tokens
                step_started = perf_counter()
                _, signed_tokens = self.step()
                step_elapsed = perf_counter() - step_started
                completed = seq.num_turn_completion_tokens - completed_before
                if signed_tokens > 0:
                    turn_metrics["prefill_tokens"] += signed_tokens
                    turn_metrics["prefill_seconds"] += step_elapsed
                    turn_metrics["prefill_generated_tokens"] += completed
                else:
                    # A final MTP verification can compute more candidate
                    # tokens than the request still has room to accept.  The
                    # chat-facing rate must therefore count the completion
                    # tokens scheduler.postprocess() actually retained, just
                    # like the benchmark command does.
                    turn_metrics["decode_tokens"] += completed
                    turn_metrics["decode_seconds"] += step_elapsed
                pbar.update(completed)
        finally:
            pbar.close()
        token_ids = list(seq.turn_completion_token_ids)
        turn_metrics["generated_tokens"] = len(token_ids)
        return {
            "text": self._decode_tokens(token_ids),
            "token_ids": token_ids,
            "metrics": turn_metrics,
        }

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
