from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
import random

class LengthPredictor:
    def __init__(self):
        pass

    def predict_one(self, seq_group: SequenceGroup) -> None:
        pass

    def predict(self, running_queue: List[SequenceGroup]) -> deque:
        '''
        inplace prediction of the length of the sequence'''
        for seq_group in running_queue:
            self.predict_one(seq_group)
        return running_queue

class RandomLength(LengthPredictor):
    def __init__(self):
        super().__init__()
        random.seed(5)

    def predict_one(self, seq_group: SequenceGroup) -> None:
        is_prefill = seq_group.is_prefill()
        if is_prefill:
            seq_group.remaining_decode = 0
        elif seq_group.remaining_decode == 0:
            # update remaining decode only when seq is just scheduled into running
            decoding_length = seq_group.get_seqs()[0].get_prompt_len()
            assert decoding_length > 0
            # pred_length = random.randint(decoding_length, decoding_length*2)
            pred_length = 2*decoding_length
            seq_group.remaining_decode = pred_length

    # def predict(self, running_queue: List[SequenceGroup]) -> deque:
    #     '''
    #     prediction of the length of the sequence'''
    #     for seq_group in running_queue:
    #         is_prefill = seq_group.is_prefill()
    #         if is_prefill:
    #             seq_group.remaining_decode = 0
    #         elif seq_group.remaining_decode == 0:
    #             # update remaining decode only when seq is just scheduled into running
    #             decoding_length = seq_group.get_seqs()[0].get_output_len()
    #             pred_length = random.randint(decoding_length, decoding_length*2)
    #             seq_group.remaining_decode = pred_length
    #     return running_queue