from collections import OrderedDict, deque

from nanovllm.config import Config
from nanovllm.backends.base import BackendExecutionResult
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # 最大同时调度的序列（请求）数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 一个batch中最多的总token数（调度最大token数）
        self.eos_token_ids = frozenset(config.eos_token_ids)
        self.enable_preemption = config.enable_preemption
        self.speculative_tokens = config.mtp_max_draft_tokens if config.enable_mtp else 0
        self.max_model_len = config.max_model_len
        self.block_size = config.kvcache_block_size  # KV Cache的block大小（每个块包含的token数）
        self.enable_session_cache = bool(getattr(config, "enable_session_cache", False))
        self.native_runtime = str(getattr(config, "backend", "")).startswith("native_")
        self.native_vulkan = getattr(config, "backend", "") == "native_vulkan"
        self.max_retained_sessions = int(getattr(config, "max_retained_sessions", 0))
        self.max_consecutive_prefill_rounds = int(
            getattr(config, "max_consecutive_prefill_rounds", 0)
        )
        # BlockManager用于管理KV cache的分配与回收，第一个参数为可分配的block数，第二个为每个block的大小
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            enable_prefix_cache=config.enable_prefix_cache,
        )
        self.waiting: deque[Sequence] = deque()  # 等待调度的序列队列
        self.running: deque[Sequence] = deque()  # 正在运行的序列队列
        self.parked: OrderedDict[int, Sequence] = OrderedDict()
        self.block_releases: list[tuple[list[int], int]] = []
        self.consecutive_prefill_rounds = 0


    def is_finished(self):
        return not self.waiting and not self.running#等待队列和运行队列都为空，则表示所有任务都执行完毕

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def resume(self, seq: Sequence):
        parked = self.parked.pop(seq.seq_id, None)
        if parked is not seq:
            raise RuntimeError("chat session is not parked in this scheduler")
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.waiting.append(seq)

    def close_parked(self, seq: Sequence):
        parked = self.parked.pop(seq.seq_id, None)
        if parked is not seq:
            raise RuntimeError("chat session is not parked in this scheduler")
        self._release_sequence(seq)

    def _speculative_tokens_for(self, seq: Sequence) -> int:
        if self.speculative_tokens == 0:
            return 0
        # A K-token verification window consumes slots through
        # position len(seq)-1+K. NativeRunner uses the same boundary test and
        # falls back to one target step when the full window would cross the
        # configured logical context.
        return self.speculative_tokens if len(seq) + self.speculative_tokens <= self.max_model_len else 0

    def schedule(self) -> tuple[list[Sequence], bool]:#调度器的核心代码，
        """
        调度器核心代码，
        主要作用：在这一轮计算中，我们要把那些请求放进来计算？
        """
        if self._should_force_decode():
            scheduled_seqs = self._schedule_decode()
            if scheduled_seqs:
                self.consecutive_prefill_rounds = 0
                return scheduled_seqs, False

        had_running_sequences = bool(self.running)
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill 一次性处理大量token
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:#等待队列有任务，且当前调度序列数小于最大序列数（是否还有空余序列可以计算）
            seq = self.waiting[0]#获取等待队列的第一个任务
            remaining = self.max_num_batched_tokens - num_batched_tokens#计算还剩下多少token可以分配，如果同时计算太多的token会报显存
            if remaining == 0:
                break
            if not seq.block_table:#空表，kvcache为空，是全新请求
                if not self._make_native_slot_available():
                    break
                num_cached_blocks = self._allocate_or_evict_sessions(seq)
                if num_cached_blocks == -1:
                    if not self.enable_preemption and not self.running and not scheduled_seqs:
                        raise RuntimeError("request exceeds the available paged KV blocks")
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                if not self._append_or_evict_sessions(seq):
                    if not self.enable_preemption and not self.running and not scheduled_seqs:
                        raise RuntimeError("request exceeds the available paged KV blocks")
                    break
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # The native Vulkan runtime owns its 64-token Mali prefill chunk
            # policy. Keep one sequence's prompt in one execution plan so it
            # preserves graph/state semantics across those internal chunks.
            if self.native_vulkan and not scheduled_seqs:
                seq.num_scheduled_tokens = num_tokens
            else:
                seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:##处理完了所有token
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            if had_running_sequences:
                self.consecutive_prefill_rounds += 1
            else:
                self.consecutive_prefill_rounds = 0
            return scheduled_seqs, True

        scheduled_seqs = self._schedule_decode()
        assert scheduled_seqs
        self.consecutive_prefill_rounds = 0
        return scheduled_seqs, False

    def _should_force_decode(self) -> bool:
        return (
            self.max_consecutive_prefill_rounds > 0
            and self.consecutive_prefill_rounds >= self.max_consecutive_prefill_rounds
            and bool(self.waiting)
            and bool(self.running)
        )

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            speculative_tokens = self._speculative_tokens_for(seq)
            if not self.enable_preemption and not self.block_manager.can_append(seq, speculative_tokens):
                raise RuntimeError("paged KV cache is full and preemption is disabled")
            while not self.block_manager.can_append(seq, speculative_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq, speculative_tokens)
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def _allocate_or_evict_sessions(self, seq: Sequence) -> int:
        """Reserve a new sequence's blocks, evicting idle sessions only if needed."""

        num_cached_blocks = self.block_manager.can_allocate(seq)
        while num_cached_blocks == -1 and self._evict_oldest_parked():
            num_cached_blocks = self.block_manager.can_allocate(seq)
        return num_cached_blocks

    def _make_native_slot_available(self) -> bool:
        """Keep long-lived parked sequences inside the runtime's slot budget."""

        if not self.native_runtime:
            return True
        while len(self.running) + len(self.parked) >= self.max_num_seqs:
            if not self._evict_oldest_parked():
                return False
        return True

    def _append_or_evict_sessions(self, seq: Sequence) -> bool:
        """Make a resumed session's block table cover its newly appended turn."""

        while not self.block_manager.can_append(seq) and self._evict_oldest_parked():
            pass
        if not self.block_manager.can_append(seq):
            return False
        self.block_manager.may_append(seq)
        return True

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        block_ids = self.block_manager.deallocate(seq)
        self.block_releases.append((block_ids, seq.seq_id))
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], result: BackendExecutionResult, is_prefill: bool):
        assert len(seqs) == len(result.token_ids)
        for seq, token_ids in zip(seqs, result.token_ids):
            assert token_ids
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            if not is_prefill:
                seq.num_cached_tokens += len(token_ids) - 1
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            for token_id in token_ids:
                seq.append_token(token_id)
                if ((not seq.ignore_eos and token_id in self.eos_token_ids) or
                        seq.num_turn_completion_tokens == seq.max_tokens):
                    self.running.remove(seq)
                    if seq.retain_cache and self.enable_session_cache:
                        self._park(seq)
                    else:
                        seq.status = SequenceStatus.FINISHED
                        block_ids = self.block_manager.deallocate(seq)
                        self.block_releases.append((block_ids, seq.seq_id))
                    break

    def _park(self, seq: Sequence):
        seq.status = SequenceStatus.PARKED
        self.parked[seq.seq_id] = seq
        self.parked.move_to_end(seq.seq_id)
        while len(self.parked) > self.max_retained_sessions:
            self._evict_oldest_parked()

    def _evict_oldest_parked(self) -> bool:
        if not self.parked:
            return False
        _, evicted = self.parked.popitem(last=False)
        self._release_sequence(evicted)
        return True

    def _release_sequence(self, seq: Sequence):
        seq.status = SequenceStatus.FINISHED
        block_ids = self.block_manager.deallocate(seq)
        self.block_releases.append((block_ids, seq.seq_id))

    def pop_block_releases(self) -> list[tuple[list[int], int]]:
        block_releases = self.block_releases
        self.block_releases = []
        return block_releases
