from collections import deque
from typing import Deque, Optional

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.metrics.arrival_time
    
class RoundRobin(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        if seq_group.metrics.first_token_time is None:
            return now - seq_group.metrics.arrival_time
        else:
            return now - seq_group.metrics.last_token_time
        # return now - seq_group.metrics.arrival_time
    
    
class StrictDecodeFirst(Policy):
    def __init__(self, max_length=30) -> None:
        self.max_length = max_length
    
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        wait_time = now - seq_group.metrics.arrival_time
        int_len = len(str(wait_time).split('.')[0])
        is_decode = int(seq_group.remaining_decode > 0)
        short_priority = max(1, self.max_length - seq_group.remaining_decode)
        sort_score = wait_time/10**int_len + short_priority * is_decode
        if hasattr(seq_group, 'just_end'):
            sort_score *= int(not seq_group.just_end)
        return sort_score

class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS, 'sdf': StrictDecodeFirst, 'rr': RoundRobin}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
