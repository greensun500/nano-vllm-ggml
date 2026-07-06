from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # 最大同时调度的序列（请求）数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 一个batch中最多的总token数（调度最大token数）
        self.eos = config.eos  # 终止token的ID（end-of-sequence）
        self.block_size = config.kvcache_block_size  # KV Cache的block大小（每个块包含的token数）
        # BlockManager用于管理KV cache的分配与回收，第一个参数为可分配的block数，第二个为每个block的大小
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()  # 等待调度的序列队列
        self.running: deque[Sequence] = deque()  # 正在运行的序列队列
        self.block_releases: list[tuple[list[int], int]] = []


    def is_finished(self):
        return not self.waiting and not self.running#等待队列和运行队列都为空，则表示所有任务都执行完毕

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:#调度器的核心代码，
        """
        调度器核心代码，
        主要作用：在这一轮计算中，我们要把那些请求放进来计算？
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill 一次性处理大量token
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:#等待队列有任务，且当前调度序列数小于最大序列数（是否还有空余序列可以计算）
            seq = self.waiting[0]#获取等待队列的第一个任务
            remaining = self.max_num_batched_tokens - num_batched_tokens#计算还剩下多少token可以分配，如果同时计算太多的token会报显存
            if remaining == 0:
                break
            if not seq.block_table:#空表，kvcache为空，是全新请求
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:##处理完了所有token
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        block_ids = self.block_manager.deallocate(seq)
        self.block_releases.append((block_ids, seq.seq_id))
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                block_ids = self.block_manager.deallocate(seq)
                self.block_releases.append((block_ids, seq.seq_id))
                self.running.remove(seq)

    def pop_block_releases(self) -> list[tuple[list[int], int]]:
        block_releases = self.block_releases
        self.block_releases = []
        return block_releases
