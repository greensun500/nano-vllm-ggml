from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nanovllm.engine.sequence import Sequence


BackendName = Literal["cuda", "llamacpp_cpu", "llamacpp_vulkan"]
PlanMode = Literal["prefill", "decode"]


@dataclass(slots=True)
class BackendExecutionPlan:
    mode: PlanMode
    input_ids: list[int]
    positions: list[int]
    seq_ids: list[int]
    scheduled_token_counts: list[int]
    block_tables: list[list[int]]
    slot_mapping: list[int]
    context_lens: list[int]
    num_cached_tokens: list[int]
    temperatures: list[float]

    @property
    def is_prefill(self) -> bool:
        return self.mode == "prefill"


def build_execution_plan(seqs: list[Sequence], is_prefill: bool, block_size: int) -> BackendExecutionPlan:
    input_ids: list[int] = []
    positions: list[int] = []
    seq_ids: list[int] = []
    scheduled_token_counts: list[int] = []
    block_tables: list[list[int]] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []
    num_cached_tokens: list[int] = []
    temperatures: list[float] = []

    if is_prefill:
        for seq in seqs:
            start = seq.num_cached_tokens
            count = seq.num_scheduled_tokens
            end = start + count
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            seq_ids.append(seq.seq_id)
            scheduled_token_counts.append(count)
            block_tables.append(list(seq.block_table))
            context_lens.append(end)
            num_cached_tokens.append(start)
            temperatures.append(seq.temperature)
            if not seq.block_table:
                continue
            start_block = start // block_size
            end_block = (end + block_size - 1) // block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * block_size
                if i == start_block:
                    slot_start += start % block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * block_size + block_size
                else:
                    slot_end = seq.block_table[i] * block_size + end - i * block_size
                slot_mapping.extend(range(slot_start, slot_end))
    else:
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            seq_ids.append(seq.seq_id)
            scheduled_token_counts.append(seq.num_scheduled_tokens)
            block_tables.append(list(seq.block_table))
            context_lens.append(len(seq))
            num_cached_tokens.append(seq.num_cached_tokens)
            temperatures.append(seq.temperature)
            slot_mapping.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)

    return BackendExecutionPlan(
        mode="prefill" if is_prefill else "decode",
        input_ids=input_ids,
        positions=positions,
        seq_ids=seq_ids,
        scheduled_token_counts=scheduled_token_counts,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        num_cached_tokens=num_cached_tokens,
        temperatures=temperatures,
    )


class BackendRunner:
    def allocate_kv_cache(self, num_blocks: int, block_size: int):
        raise NotImplementedError

    def run(self, plan: BackendExecutionPlan) -> list[int]:
        raise NotImplementedError

    def release_blocks(self, block_ids: list[int], seq_ids: list[int] | None = None):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError
