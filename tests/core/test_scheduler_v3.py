from typing import List
from unittest.mock import MagicMock

import pytest  # noqa

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
# from vllm.core.scheduler import Scheduler
from vllm.core.scheduler_v2 import SchedulerV3 as Scheduler 
from vllm.core.scheduler_v2 import knapsack
from vllm.sequence import Logprob, SequenceGroup

from .utils import create_dummy_prompt


def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]


def append_new_token(seq_group, token_id: int):
    for seq in seq_group.get_seqs():
        seq.append_token_id(token_id, {token_id: Logprob(token_id)})


def schedule_and_update_computed_tokens(scheduler):
    metas, out = scheduler.schedule()
    for s, meta in zip(out.scheduled_seq_groups, metas):
        s.seq_group.update_num_computed_tokens(meta.token_chunk_size)
    return metas, out

def test_dp():
    N = 4
    M = 10
    weights = [2,3,4,8]
    max_weights = [8,1,1,2]
    knapsack(weights, max_weights, N, M)