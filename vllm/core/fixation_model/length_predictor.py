from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
import random, time
import numpy as np
from vllm.core.fixation_model.eye_attention import Eyettention, tokens_to_inputs

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
    
    def assign_one(self, seq_group: SequenceGroup, mode = None) -> None:
        seq = seq_group.get_seqs()[0]
        # is_prefill = seq_group.is_prefill()
        if seq.tokens is None or mode == "prompt":
            self._predict_by_prompt_len(seq_group)
        else:
            self._predict_by_output(seq_group)

    def _predict_by_prompt_len(self, seq_group: SequenceGroup) -> None:
        raise NotImplementedError

    def _predict_by_output(self, seq_group: SequenceGroup) -> None:
        raise NotImplementedError


class RandomLength:
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

    def assign_one(self, seq_group: SequenceGroup, mode: str = None) -> None:
        # update remaining decode only when seq is just scheduled into running
        decoding_length = seq_group.get_seqs()[0].get_prompt_len()
        assert decoding_length > 0
        # pred_length = random.randint(decoding_length, decoding_length*2)
        pred_length = int(1.0*decoding_length)
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

class ModelPredictorV0(RandomLength):
    def __init__(self, max_sequence_length: int = 10):
        super().__init__()
        self.max_sequence_length: int = max_sequence_length
        self.model = Eyettention(max_sequence_length=max_sequence_length)

    def _predict_by_output(self, seq_group: SequenceGroup) -> None:
        # update remaining decode only when seq is just scheduled into running
        seq = seq_group.get_seqs()[0]
        output_ids = seq.data.output_token_ids
        output_text = seq.tokens[-len(output_ids):]

        token_ids, token_lengths = tokens_to_inputs(output_text, output_ids, self.max_sequence_length)
        pred_length = self.model.predict(token_ids, token_lengths).item()

        seq_group.remaining_decode = pred_length

class ModelPredictorV1(RandomLength):
    def __init__(self, max_sequence_length: int = 10):
        self.max_sequence_length: int = max_sequence_length
        self.model = Eyettention(max_sequence_length=max_sequence_length)

    def _predict_by_output(self, seq_group: SequenceGroup) -> None:
        # update remaining decode only when seq is just scheduled into running
        seq = seq_group.get_seqs()[0]
        output_ids = seq.get_output_token_ids()
        output_text = seq.tokens[7:] # 7 for "_GPT_4_Correct_Assistant:"
        assert len(output_ids) == len(output_text)

        t0 = time.time()
        token_ids, token_lengths = tokens_to_inputs(output_text, output_ids, self.max_sequence_length)
        t1 = time.time()
        pred_length = self.model.predict(token_ids, token_lengths)
        print(f"Time for prediction: {time.time()-t1}, time for data: {t1-t0}")
        assert pred_length > 0

        seq_group.remaining_decode = pred_length

class ModelPredictor(RandomLength):
    def __init__(self, max_sequence_length: int = 10):
        super().__init__()
        self.max_sequence_length: int = max_sequence_length
        self.model = Eyettention(max_sequence_length=max_sequence_length)

    def _predict_by_output(self, seq_group: SequenceGroup) -> None:
        # update remaining decode only when seq is just scheduled into running
        seq = seq_group.get_seqs()[0]
        # output_ids = seq.get_output_token_ids()
        output_text = "".join(seq.tokens[7:]) # 7 for "_GPT_4_Correct_Assistant:"

        t0 = time.time()
        token_ids, token_lengths = self.model.text_to_inputs(output_text, self.max_sequence_length)
        t1 = time.time()
        pred_length = self.model.predict(token_ids, token_lengths)
        print(f"Time for prediction: {time.time()-t1}, time for data: {t1-t0}")

        seq_group.remaining_decode = pred_length

    def assign_one(self, seq_group: SequenceGroup, mode = None) -> None:
        if mode == "prompt":
            super().assign_one(seq_group)
        else:
            seq = seq_group.get_seqs()[0]
            # output_text = "".join(seq.tokens[7:]) # 7 for "_GPT_4_Correct_Assistant:"
            output_text = seq.prompt + seq.output_text

            # t0 = time.time()
            token_ids, token_lengths = self.model.text_to_inputs(output_text, self.max_sequence_length)
            # t1 = time.time()
            pred_length = self.model.predict(token_ids, token_lengths)
            # print(f"Time for prediction: {time.time()-t1}, time for data: {t1-t0}")

            seq_group.remaining_decode = max(pred_length, 10)