from collections import deque
from typing import Deque

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
    
class DecodeFirst(Policy):
    
        def get_priority(
            self,
            now: float,
            seq_group: SequenceGroup,
        ) -> float:
            wait_time = now - seq_group.metrics.arrival_time
            int_len = len(str(wait_time).split('.')[0])
            is_decode = int(seq_group.remaining_decode > 0)
            sort_score = wait_time/10**int_len + is_decode
            return sort_score

class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS, 'df': DecodeFirst}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
