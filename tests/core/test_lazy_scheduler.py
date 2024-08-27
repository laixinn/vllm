from typing import List
from unittest.mock import MagicMock
import copy

import pytest  # noqa

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
# from vllm.core.scheduler import Scheduler
from vllm.core.scheduler_v2 import LazyScheduler as Scheduler
from vllm.core.scheduler_v2 import LazyStates
from vllm.sequence import Logprob, SequenceGroup, SequenceStatus

from .utils import create_dummy_prompt

from vllm.core.scheduler_v2 import LazySequenceGroup

def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]


def append_new_token(seq_group, token_id: int):
    for seq in seq_group.get_seqs():
        seq.append_token_id(token_id, {token_id: Logprob(token_id)})


def schedule_and_update_computed_tokens(scheduler):
    scheduler.lazy_schedule()
    metas, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    scheduler.sync_running()
    return metas, out

def test_class_convert():
    block_size = 4
    _, seq_group = create_dummy_prompt(str(0), prompt_length=block_size)
    lazy_seq_group = LazySequenceGroup(seq_group)
    # test inherite
    append_new_token(seq_group, 1)
    assert id(lazy_seq_group.get_seqs()[0].logical_token_blocks) == \
        id(seq_group.get_seqs()[0].logical_token_blocks)
    seq_group.request_id = '1'
    assert seq_group.request_id == lazy_seq_group.request_id
    # test convert to base class
    lazy_seq_group.new_seq_states[0] = SequenceStatus.SWAPPED
    assert len(lazy_seq_group.get_seqs(SequenceStatus.SWAPPED)) == 1
    lazy_seq_group = lazy_seq_group.seq_group
    assert len(lazy_seq_group.get_seqs(SequenceStatus.SWAPPED)) == 0, f"{lazy_seq_group.get_seqs()}"

    new_seq_group = LazySequenceGroup(lazy_seq_group)
    ano_seq_group = LazySequenceGroup(lazy_seq_group)
    new_seq_group.lazy_state = LazyStates.SWAP
    ano_seq_group.lazy_state = LazyStates.RECOMPUTE
    assert ano_seq_group.lazy_state != new_seq_group.lazy_state

def test_simple():
    """Verify basic scheduling works."""
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       num_seq_group,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Schedule seq groups prompts.
    num_tokens = block_size * num_seq_group
    scheduler.lazy_schedule()
    # seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    seq_group_meta, out = scheduler.schedule()
    scheduler.lazy_schedule()
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_tokens
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group
    assert all([s.is_prefill() for s in running])
    assert all([not s.is_prefill() for s in scheduler.running])
    for s in running:
        append_new_token(s, 1)
    for s, meta in zip(out.scheduled_seq_groups, seq_group_meta):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)

    # Schedule seq groups generation.
    # seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    seq_group_meta, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, seq_group_meta):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    assert set(get_sequence_groups(out)) == set(running)
    assert out.num_batched_tokens == num_seq_group
    assert (not out.blocks_to_copy and not out.blocks_to_swap_in
            and not out.blocks_to_swap_out)
    assert len(seq_group_meta) == num_seq_group
    # All seq groups are in decoding phase, remaining decode >= 1
    assert all([s.remaining_decode >= 1 for s in running])

def test_pipeline():
    """Verify basic scheduling works."""
    block_size = 4
    num_seq_group = 4
    max_model_len = 16
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       num_seq_group,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    scheduler2 = Scheduler(scheduler_config, cache_config, None, queue_size=2) # pipeline
    running: List[SequenceGroup] = []
    running2: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(num_seq_group):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

        _, seq_group2 = create_dummy_prompt(str(i), prompt_length=block_size)
        scheduler2.add_seq_group(seq_group2)
        running2.append(seq_group2)

    # Schedule seq groups prompts.
    num_tokens = block_size * num_seq_group
    seq_group_meta11, out11 = schedule_and_update_computed_tokens(scheduler)
    for s in running:
        append_new_token(s, 1)
    seq_group_meta21, out21 = schedule_and_update_computed_tokens(scheduler)

    scheduler2.lazy_schedule()
    scheduler2.lazy_schedule()
    seq_group_meta12, out12 = scheduler2.schedule()
    for s, meta in zip(out12.scheduled_seq_groups, seq_group_meta12):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    for s in running2:
        append_new_token(s, 1)
    seq_group_meta22, out22 = scheduler2.schedule()

    for meta1, meta2 in zip(seq_group_meta11, seq_group_meta12):
        for key in vars(meta1):
            assert str(meta1.__dict__[key]) == str(meta2.__dict__[key])

    for meta1, meta2 in zip(seq_group_meta21, seq_group_meta22):
        for key in vars(meta1):
            assert str(meta1.__dict__[key]) == str(meta2.__dict__[key])

    assert str(out11) == str(out12)
    assert str(out21) == str(out22)

def test_chunk():
    """Verify prefills are chunked properly."""
    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Verify the second request is chunked.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 60
    # Verify it is chunked.
    assert seq_group_meta[1].token_chunk_size == 4
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    # Only the first seq group has a new token appended.
    append_new_token(running[0], 1)

    # One chunked prefill, and one decoding.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert set(get_sequence_groups(out)) == set(running)
    # The first one is prefill. Scheduler guarantees ordering.
    assert seq_group_meta[0].token_chunk_size == 56
    # The second one is a chunked prefill.
    assert seq_group_meta[1].token_chunk_size == 1
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 57
    # check remaining decode
    assert running[0].remaining_decode >= 1


def test_complex():
    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
        assert seq_group.is_prefill()

    # Verify the second request is chunked.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)

    assert set(get_sequence_groups(out)) == set(running)
    assert seq_group_meta[0].token_chunk_size == 60
    # Verify it is chunked.
    assert seq_group_meta[1].token_chunk_size == 4
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    # Only the first seq group has a new token appended.
    append_new_token(running[0], 1)

    # Add 2 more requsets.
    for i in range(2, 4):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=60)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Decoding & chunked prefill & first chunk of 3rd request is scheduled.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 3
    # The first one is the first chunked prefill.
    assert seq_group_meta[0].token_chunk_size == 7
    # The second one is the second new chunked prefill.
    assert seq_group_meta[1].token_chunk_size == 56
    # The last one is decode.
    assert seq_group_meta[2].token_chunk_size == 1
    # Two of them are in chunked prefill.
    assert out.num_prefill_groups == 2
    assert out.num_batched_tokens == 64
    # decoding phase should not have ramaining decode = 0
    assert running[0].remaining_decode >= 1
    # The first 2 requests are now in decodine phase.
    append_new_token(running[0], 1)
    assert not running[0].is_prefill()
    append_new_token(running[1], 1)
    assert not running[1].is_prefill()
    # The third request is still in prefill stage.
    assert running[2].is_prefill()

    r0decode = running[0].remaining_decode
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert running[0].remaining_decode == r0decode - 1
    assert running[1].remaining_decode >= 1


def test_maximal_decoding():
    """Verify decoding requests are prioritized."""
    block_size = 4
    max_seqs = 2
    max_model_len = 2
    max_num_batched_tokens = 2
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    for i in range(2):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=2)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)
        assert seq_group.is_prefill()

    # The first prefill is scheduled.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 1
    assert seq_group_meta[0].token_chunk_size == 2
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    # Only the first seq group has a new token appended.
    append_new_token(running[0], 1)

    # Create one more seq_group.
    _, seq_group = create_dummy_prompt("3", prompt_length=2)
    scheduler.add_seq_group(seq_group)
    running.append(seq_group)
    assert seq_group.is_prefill()
    # The first decoding + second chunk is scheduled.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    assert running[2].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    assert running[1].get_num_uncomputed_tokens() == 1
    assert running[0].remaining_decode >= 1
    running[0].remaining_decode = 3
    append_new_token(running[0], 1)

    # Decoding + running prefill is prioritized.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    assert not running[1].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    assert running[0].remaining_decode == 2
    append_new_token(running[0], 1)
    append_new_token(running[1], 1)

    # Decoding + running prefill is prioritized.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    assert not running[1].is_prefill()
    assert out.num_prefill_groups == 0
    assert out.num_batched_tokens == 2
    assert running[0].remaining_decode == 1
    assert running[1].remaining_decode >= 1
    running[1].remaining_decode = 3
    append_new_token(running[0], 1)
    append_new_token(running[1], 1)

    # seq 0 is swapped out due to running out of remaining decode.
    # scheduler.abort_seq_group(running[0].request_id)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[1].is_prefill()
    assert running[2].is_prefill()
    # NOTE: strict policy
    # assert out.num_prefill_groups == 1
    # NOTE: non-strict
    assert out.num_prefill_groups == 0

    assert out.num_batched_tokens == 2
    # assert scheduler.swapped[0].request_id == "0"
    assert running[1].remaining_decode == 2
    assert running[0].remaining_decode == 0
    # assert not running[0].is_prefill()
    # NOTE: strict policy
    # assert running[2].get_num_uncomputed_tokens() == 1
    # NOTE: non-strict
    assert running[2].get_num_uncomputed_tokens() == 2


def test_prompt_limit():
    """Verify max_num_batched_tokens < max_model_len is possible."""
    block_size = 4
    max_seqs = 32
    max_model_len = 64
    max_num_batched_tokens = 32
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    _, seq_group = create_dummy_prompt("1", prompt_length=48)
    scheduler.add_seq_group(seq_group)
    running.append(seq_group)
    assert seq_group.is_prefill()

    # The prompt length > max_num_batched_tokens should be still scheduled.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 1
    assert seq_group_meta[0].token_chunk_size == 32
    assert running[0].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 32


def test_prompt_limit_exceed():
    block_size = 4
    max_seqs = 64
    max_model_len = 32
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    _, seq_group = create_dummy_prompt("2", prompt_length=48)
    scheduler.add_seq_group(seq_group)
    running.append(seq_group)
    assert seq_group.is_prefill()
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.ignored_seq_groups) == 1
    assert out.ignored_seq_groups[0] == seq_group


def test_swap():
    """Verify swapping works with chunked prefill requests"""
    block_size = 4
    max_seqs = 30
    max_model_len = 200
    max_num_batched_tokens = 30
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
    scheduler.add_seq_group(seq_group)
    _, out = schedule_and_update_computed_tokens(scheduler)
    # The request is chunked.
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert seq_group.is_prefill()
    assert out.num_batched_tokens == max_num_batched_tokens

    # The last request should be swapped out.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots=0):
        return seq_group.request_id != "1"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)

    scheduler._maybe_can_append_slots = MagicMock()
    scheduler._maybe_can_append_slots.side_effect = (
        cannot_append_second_group)

    # The running prefill is now swapped.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 0
    assert out.num_batched_tokens == 0
    assert out.blocks_to_swap_out != []
    assert out.blocks_to_swap_in == []

    # Add 1 more task. Swap should be prioritized over new prefill.
    _, seq_group = create_dummy_prompt("2", prompt_length=60)
    scheduler.add_seq_group(seq_group)
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    # 3 decodes. It is swapped in.
    assert out.num_batched_tokens == 30
    assert out.blocks_to_swap_in != []
    assert out.blocks_to_swap_out == []


def test_running_prefill_prioritized_over_swap():
    block_size = 4
    max_seqs = 30
    max_model_len = 200
    max_num_batched_tokens = 30
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    _, seq_group = create_dummy_prompt("1", prompt_length=60, best_of=2)
    scheduler.add_seq_group(seq_group)
    _, out = schedule_and_update_computed_tokens(scheduler)
    # The request is chunked.
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert seq_group.is_prefill()
    assert out.num_batched_tokens == max_num_batched_tokens

    # The request should be swapped out.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots=0):
        return seq_group.request_id != "1"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)
    
    scheduler._maybe_can_append_slots = MagicMock()
    scheduler._maybe_can_append_slots.side_effect = (
        cannot_append_second_group)

    # The running prefill is now swapped.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 0
    assert out.num_batched_tokens == 0
    assert out.blocks_to_swap_out != []
    assert out.blocks_to_swap_in == []

    # Add 1 more task. Swap is not possible, so prefill is running.
    scheduler.block_manager.can_swap_in = MagicMock()
    scheduler.block_manager.can_swap_in.return_value = AllocStatus.LATER

    scheduler._maybe_can_swap_in = MagicMock()
    scheduler._maybe_can_swap_in.return_value = AllocStatus.LATER

    _, seq_group2 = create_dummy_prompt("2", prompt_length=60)
    scheduler.add_seq_group(seq_group2)
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    # 3 decodes. It is swapped in.
    assert out.num_batched_tokens == 30
    assert out.blocks_to_swap_in == []
    assert out.blocks_to_swap_out == []
    assert out.scheduled_seq_groups[0].seq_group == seq_group2

    # Now although swap is possible, running prefill is prioritized.
    scheduler.block_manager.can_swap_in.return_value = AllocStatus.OK
    scheduler._maybe_can_swap_in.return_value = AllocStatus.OK
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    # 3 decodes. It is swapped in.
    assert out.num_batched_tokens == 30
    assert out.blocks_to_swap_in == []
    assert out.blocks_to_swap_out == []
    assert not seq_group2.is_prefill()
    assert out.scheduled_seq_groups[0].seq_group == seq_group2
    append_new_token(seq_group2, 1)

    # Decoding is prioritized.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    # 3 decodes. It is swapped in.
    assert out.num_batched_tokens == 1
    assert out.blocks_to_swap_in == []
    assert out.blocks_to_swap_out == []
    assert not seq_group2.is_prefill()
    assert out.scheduled_seq_groups[0].seq_group == seq_group2
    append_new_token(seq_group2, 1)

    # Since we abort the sequence group, we can finally swap.
    scheduler.abort_seq_group(seq_group2.request_id)
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_batched_tokens == 30
    assert out.blocks_to_swap_in != []
    assert out.blocks_to_swap_out == []


def test_chunked_prefill_preempt():
    """Verify preempt works with chunked prefill requests"""
    block_size = 4
    max_seqs = 30
    max_model_len = 200
    max_num_batched_tokens = 30
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)

    _, seq_group = create_dummy_prompt("1", prompt_length=60)
    scheduler.add_seq_group(seq_group)
    _, out = schedule_and_update_computed_tokens(scheduler)
    # The request is chunked.
    # prefill scheduled now.
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert seq_group.is_prefill()
    assert out.num_batched_tokens == max_num_batched_tokens

    # The request should be preempted.
    scheduler.block_manager.can_append_slots = MagicMock()

    def cannot_append_second_group(seq_group, num_lookahead_slots=0):
        return seq_group.request_id != "1"

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)

    scheduler._maybe_can_append_slots = MagicMock()
    scheduler._maybe_can_append_slots.side_effect = (
        cannot_append_second_group)

    # The running prefill is now preempted.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 0
    assert out.num_batched_tokens == 0
    assert out.blocks_to_swap_out == []
    assert out.blocks_to_swap_in == []

    # Make sure we can reschedule preempted request.
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert seq_group.is_prefill()
    assert out.num_batched_tokens == max_num_batched_tokens
    assert seq_group.get_num_uncomputed_tokens() == 30

    # We should be able to run prefill twice as it is chunked.
    def cannot_append_second_group(seq_group, num_lookahead_slots=0):
        return True

    scheduler.block_manager.can_append_slots.side_effect = (
        cannot_append_second_group)
    scheduler._maybe_can_append_slots.side_effect = (
        cannot_append_second_group)
    _, out = schedule_and_update_computed_tokens(scheduler)
    assert len(out.scheduled_seq_groups) == 1
    assert out.num_prefill_groups == 1
    assert not seq_group.is_prefill()
    assert out.num_batched_tokens == max_num_batched_tokens


def test_chunked_prefill_max_seqs():
    block_size = 4
    max_seqs = 2
    max_model_len = 80
    max_num_batched_tokens = 64
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 8
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running = []

    _, seq_group = create_dummy_prompt("1", prompt_length=65)
    scheduler.add_seq_group(seq_group)
    running.append(seq_group)
    # The first prefill is chunked.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert seq_group_meta[0].token_chunk_size == max_num_batched_tokens
    assert len(get_sequence_groups(out)) == 1

    # Add new requests.
    for i in range(4):
        _, seq_group = create_dummy_prompt(str(i), prompt_length=65)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # Make sure only 2 requests are scheduled.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert out.num_batched_tokens == max_num_batched_tokens
    assert len(get_sequence_groups(out)) == 2
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    append_new_token(running[0], 1)

    # Although we have enough token budget, we can only schedule max_seqs.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert seq_group_meta[0].token_chunk_size == 2
    assert seq_group_meta[1].token_chunk_size == 1
    assert out.num_batched_tokens == 3
    assert len(get_sequence_groups(out)) == max_seqs
    assert not running[0].is_prefill()
    assert not running[1].is_prefill()


def test_maximal_decoding_priority():
    # LogicalTokenBlock is controlled in create_dummy_prompt
    # block_size=prompt_length is neccesaary, 
    # prefilling needs preempt the whole prompt
    block_size = 2
    max_seqs = 2
    max_model_len = 2
    max_num_batched_tokens = 2
    scheduler_config = SchedulerConfig(max_num_batched_tokens,
                                       max_seqs,
                                       max_model_len,
                                       enable_chunked_prefill=True)
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 4
    cache_config.num_gpu_blocks = 4
    scheduler = Scheduler(scheduler_config, cache_config, None)
    running: List[SequenceGroup] = []

    # Add seq groups to scheduler.
    prompt_length_list = [2, 1, 4]
    for i, prompt_length in enumerate(prompt_length_list):
        _, seq_group = create_dummy_prompt(str(i), 
                                           prompt_length=prompt_length)
                                        #    block_size=block_size)
        scheduler.add_seq_group(seq_group)
        running.append(seq_group)

    # *1: Schedule seq groups prompts.
    new_token_cnt = len(prompt_length_list)
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(get_sequence_groups(out)) == 1
    assert seq_group_meta[0].token_chunk_size == 2
    assert not running[0].is_prefill()
    assert running[1].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    # cache for seq 0 prompt, check 
    # scheduler.block_manager.get_block_table(running[0].get_seqs()[0])
    assert len(scheduler.block_manager.get_block_table(running[0].get_seqs()[0])) == 1
    assert scheduler.block_manager.gpu_allocator.get_num_free_blocks() == 3
    # Only the first seq group has a new token appended.
    append_new_token(running[0], new_token_cnt)
    new_token_cnt += 1

    # *2: seq 0 decode, seq 1 decode
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    # add one for seq 0 decode, add one for seq 1 prompt
    assert len(scheduler.block_manager.get_block_table(running[0].get_seqs()[0])) == 2
    assert len(scheduler.block_manager.get_block_table(running[1].get_seqs()[0])) == 1
    assert scheduler.block_manager.gpu_allocator.get_num_free_blocks() == 1
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    # assert not running[1].is_prefill()
    assert running[2].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    assert running[0].remaining_decode >= 1
    running[0].remaining_decode = 3
    append_new_token(running[0], new_token_cnt)
    new_token_cnt += 1
    append_new_token(running[1], new_token_cnt)
    new_token_cnt += 1

    # *3: initialize seq 1 remaining decode, running out of cache
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    # add one for seq 1 decode
    assert len(scheduler.block_manager.get_block_table(running[0].get_seqs()[0])) == 2
    assert len(scheduler.block_manager.get_block_table(running[1].get_seqs()[0])) == 2
    assert scheduler.block_manager.gpu_allocator.get_num_free_blocks() == 0
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    assert not running[1].is_prefill()
    assert out.num_prefill_groups == 0
    assert out.num_batched_tokens == 2
    assert running[0].remaining_decode == 2
    assert running[1].remaining_decode >= 1
    running[1].remaining_decode = 2
    append_new_token(running[0], new_token_cnt)
    new_token_cnt += 1
    append_new_token(running[1], new_token_cnt)
    new_token_cnt += 1

    # *4: seq 1 should be swapped out due to insufficient cache for seq 0 decoding
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(scheduler.block_manager.get_block_table(running[0].get_seqs()[0])) == 3
    # seq 1 is swapped out to cpu cache, but has a pointer to seq 1 cache,
    # due to their prompt is same.
    assert len(scheduler.block_manager.get_block_table(running[1].get_seqs()[0])) == 2
    assert len(scheduler.block_manager.get_block_table(running[2].get_seqs()[0])) == 1
    assert scheduler.block_manager.gpu_allocator.get_num_free_blocks() == 0
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[0].is_prefill()
    assert not running[1].is_prefill()
    assert running[2].is_prefill()
    assert out.num_batched_tokens == 2
    assert scheduler.swapped[0].request_id == "1"
    assert scheduler.running[0].request_id == "2"
    assert running[1].remaining_decode == 1
    assert running[0].remaining_decode == 1
    assert running[2].get_num_uncomputed_tokens() == 3

    # *5: seq 0 is out, seq 2 is scheduled
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    # seq 2 prefilling preempt 2 blocks for 4 tokens, although only 2 is computed
    assert scheduler.block_manager.gpu_allocator.get_num_free_blocks() == 3
    assert len(scheduler.block_manager.get_block_table(running[2].get_seqs()[0])) == 1
    assert len(get_sequence_groups(out)) == 1
    assert seq_group_meta[0].token_chunk_size == 2
    assert not running[1].is_prefill()
    assert running[2].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    assert running[1].remaining_decode == 1
    assert running[0].remaining_decode == 0
    assert running[2].get_num_uncomputed_tokens() == 1
    assert running[1].get_num_uncomputed_tokens() == 1

    # *6: seq 1 is in, and prioritize allocating 3 blocks to seq 1's swap-in.
    seq_group_meta, out = schedule_and_update_computed_tokens(scheduler)
    assert len(scheduler.block_manager.get_block_table(running[1].get_seqs()[0])) == 3
    assert len(scheduler.block_manager.get_block_table(running[2].get_seqs()[0])) == 1
    assert scheduler.block_manager.gpu_allocator.get_num_free_blocks() == 0
    assert len(get_sequence_groups(out)) == 2
    assert seq_group_meta[0].token_chunk_size == 1
    assert seq_group_meta[1].token_chunk_size == 1
    assert not running[1].is_prefill()
    assert not running[2].is_prefill()
    assert out.num_prefill_groups == 1
    assert out.num_batched_tokens == 2
    assert running[1].remaining_decode == 1
    assert running[2].get_num_uncomputed_tokens() == 0
    assert running[1].get_num_uncomputed_tokens() == 0