import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm

from nanovllm.config import Config
from nanovllm.backends import create_backend
from nanovllm.backends.base import build_execution_plan
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


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
        if config.tokenizer_backend == "llamacpp":
            self.tokenizer = None
            config.eos_token_ids = self.model_runner.call("eog_token_ids")
        else:
            from transformers import AutoTokenizer

            tokenizer_path = config.tokenizer or config.model
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            eos_token_id = self.tokenizer.eos_token_id
            config.eos_token_ids = (eos_token_id,) if isinstance(eos_token_id, int) else tuple(eos_token_id or ())
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
        if isinstance(prompt, str):
            if self.config.tokenizer_backend == "llamacpp":
                prompt = self.model_runner.call("tokenize", prompt)
            else:
                prompt = self.tokenizer.encode(prompt)#把prompt转换为tokenizer期望的格式
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def flush_backend_releases(self):
        for block_ids, seq_id in self.scheduler.pop_block_releases():
            self.model_runner.call("release_blocks", block_ids, [seq_id])

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()#调度返回这一轮要计算的seq列表（包括prompt+采样参数），以及是否是prefill
        self.flush_backend_releases()
        plan = build_execution_plan(seqs, is_prefill, self.config.kvcache_block_size)
        result = self.model_runner.call("run", plan)
        num_tokens = (sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill
                      else -sum(len(token_ids) for token_ids in result.token_ids))#计算这轮处理了多少token数据
        self.scheduler.postprocess(seqs, result, is_prefill)
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
        if self.config.tokenizer_backend == "llamacpp":
            outputs = [{"text": self.model_runner.call("detokenize", token_ids), "token_ids": token_ids} for token_ids in outputs]
        else:
            outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
