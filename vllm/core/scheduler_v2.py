from __future__ import annotations

import enum, copy
from itertools import chain
from queue import Queue
import threading

from vllm.core.scheduler import *
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import SequenceGroup, SequenceStage

TEST = 1

class LazyStates(enum.Enum):
    RECOMPUTE = enum.auto()
    SWAP = enum.auto()
    PREFILL = enum.auto()
    DECODE = enum.auto()
    NEVER = enum.auto()
    IGNORE = enum.auto()

@dataclass
class LazySchedule:
    seq_group: SequenceGroup
    state: LazyStates
    old_states: List[SequenceStatus]
    num_running_tokens: int = 0

class SchedulerV2(Scheduler):
    def __init__(self, 
                scheduler_config: SchedulerConfig,
                cache_config: CacheConfig,
                lora_config: Optional[LoRAConfig]) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        # track schedule gap
        self.last_schedule_time: float = 0.0
        # length predictor
        self.length_predictor = RandomLength()

        self.cache_consumption = 0

    def _schedule(self) -> SchedulerOutputs:
        return self._schedule_chunked_prefill_with_predicted_length()
    
    def _schedule_chunked_prefill_with_predicted_length(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        total_seq_groups = len(self.waiting) + len(self.running) + len(self.swapped)

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                    SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        gpu_usage = self.block_manager.gpu_allocator.get_num_free_blocks() / self.block_manager.gpu_allocator.get_num_total_blocks()
        if gpu_usage > 0.9:
            self.running, decode_preempted = self._update_running_decode(
                self.running,
                budget,
                curr_loras,
                enable_chunking=True)
        else:
            self._update_remaining_decode(self.running)
            decode_preempted = SchedulerRunningOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        # fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        df_policy = PolicyFactory.get_policy(policy_name="sdf")
        remaining_running, running_scheduled = self._schedule_running(
            self.running,
            budget,
            curr_loras,
            df_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) + len(
                    decode_preempted.swapped_out) + len(
                        decode_preempted.preempted) == 0:
            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, df_policy)

        # Schedule new prefills.
        remaining_waiting, prefills = self._schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        self.waiting.extendleft(decode_preempted.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        self.swapped.extend(decode_preempted.swapped_out)
        # update metrics
        assert total_seq_groups == len(self.waiting) + len(self.running) + len(self.swapped) + len(prefills.ignored_seq_groups)
        self._update_time_metrcis(prefills, running_scheduled, 
                                swapped_in, decode_preempted)

        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + 
                            decode_preempted.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
                        swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) + 
                    len(decode_preempted.preempted)),
        )
    
    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.

        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)
        # running_queue = self.running

        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                t0 = time.time()
                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()
                    # self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
                    # fork_victim_seq_group = copy.deepcopy(victim_seq_group)
                    # self._lazy_preempt(fork_victim_seq_group)
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                    self.cache_consumption += time.time() - t0
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    # self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
                    # fork_seq_group = copy.deepcopy(seq_group)
                    # self._lazy_preempt(fork_seq_group)
                    preempted_mode = self._preempt(seq_group,
                                                   blocks_to_swap_out)
                    # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    self.cache_consumption += time.time() - t0
                    break
            else:
                t0 = time.time()
                self._append_slots(seq_group, blocks_to_copy)
                self.cache_consumption += time.time() - t0
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.add(seq_group.lora_int_id)

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))
        
    def _schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerSwappedInOutputs]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)
        # swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            alloc_status = self.block_manager.can_swap_in(seq_group)
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            t0 = time.time()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            self.cache_consumption += time.time() - t0
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return swapped_queue, SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []

        waiting_queue = deque([s for s in waiting_queue])

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()

            t0 = time.time()
            # self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
            # fork_seq_group = copy.deepcopy(seq_group)
            # self._lazy_allocate_and_set_running(fork_seq_group)
            self._allocate_and_set_running(seq_group)
            # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()
            self.cache_consumption += time.time() - t0

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.remaining_decode >= 1:
                # prioritize remaining decode
                preemption_mode = PreemptionMode.SWAP
            elif seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode
    
    def _predict_length(self, scheduled_seq_groups: List[ScheduledSequenceGroup]) -> List[ScheduledSequenceGroup]:
        list_seq_group = [sch_seq_group.seq_group for sch_seq_group in scheduled_seq_groups]
        list_seq_group = self.length_predictor.predict(list_seq_group)
        for updated_seq_group, sch_seq_group in zip(list_seq_group, scheduled_seq_groups):
            sch_seq_group.seq_group = updated_seq_group
        return scheduled_seq_groups

    def _update_running_decode(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        '''
        preempt all running seq_group
        '''
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []
        remaining_running: Deque[Sequence] = deque()

        while running_queue:
            seq_group = running_queue[0]
            #     # =1: finish current decoding, swap out
            #     # >1: current decoding
            #     # =0 and is_prefill=1: prefilling
            #     # =0 and is_prefill=0: decoding first token
            if seq_group.remaining_decode == 1:
                seq_group.remaining_decode = 0
            else:
                if seq_group.remaining_decode > 1:
                    seq_group.remaining_decode -= 1
                elif seq_group.is_prefill() and seq_group.remaining_decode == 0:
                    assert seq_group.remaining_decode == 0
                elif not seq_group.is_prefill() and seq_group.remaining_decode == 0:
                    self.length_predictor.predict_one(seq_group)
                    assert seq_group.remaining_decode >= 1
                else:
                    assert False, "remaining_decode out of expected cases"
                
                remaining_running.append(seq_group)
                running_queue.popleft()
                continue

            # TODO: check this, seem to have bugs
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()

            budget.subtract_num_batched_tokens(seq_group.request_id,
                                                num_running_tokens)
            num_running_seqs = seq_group.get_max_num_running_seqs()
            budget.subtract_num_seqs(seq_group.request_id,
                                        num_running_seqs)
            if curr_loras is not None and seq_group.lora_int_id > 0:
                curr_loras.remove(seq_group.lora_int_id)
                
            # No other sequence groups can be preempted.
            # Preempt the current sequence group.
            # NOTE: use SWAP because RECOMPUTE loses remaining decode
            # but remaining_decode=0, so finally use RECOMPUTE
            preempted_mode = self._preempt(seq_group,
                                            blocks_to_swap_out)
            if preempted_mode == PreemptionMode.RECOMPUTE:
                preempted.append(seq_group)
            else:
                swapped_out.append(seq_group)

        return remaining_running, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))
    
    def _update_remaining_decode(
        self,
        running_queue: deque,
    ) -> None:
        for seq_group in running_queue:
            #     # =1: finish current decoding, swap out
            #     # >1: current decoding
            #     # =0 and is_prefill=1: prefilling
            #     # =0 and is_prefill=0: decoding first token
            if seq_group.remaining_decode == 1:
                seq_group.remaining_decode = 0
                seq_group.just_end = 1
            else:
                if seq_group.remaining_decode > 1:
                    seq_group.remaining_decode -= 1
                elif seq_group.is_prefill() and seq_group.remaining_decode == 0:
                    assert seq_group.remaining_decode == 0
                elif not seq_group.is_prefill() and seq_group.remaining_decode == 0:
                    self.length_predictor.predict_one(seq_group)
                    seq_group.just_end = 0
                    assert seq_group.remaining_decode >= 1
                else:
                    assert False, "remaining_decode out of expected cases"

import math
from numba import jit
# from numba.typed import List
import numba
# @jit(nopython=True)
def knapsackV0(
        weights: List[int], max_weights: List[int], N: int, # N items, their weights, and max weight growth
        M: int, # capacity
    ) -> Tuple[int, List[List[int]]]: # T, and selected items for each T

    def best_of(values: List[int]) -> Tuple[int, int]:
        max_value = max(values)
        max_index = values.index(max_value)
        return max_value, max_index
    
    if N == 1:
        best_T = max_weights[0] + 1
        selected_items = [[1] for _ in range(best_T+1)]
        selected_items[0][0] = 0
        return best_T, selected_items
        
    # final_score = sum(weights+max_weights)
    final_score = sum(weights) + sum(max_weights)
    max_T = int(sum(max_weights)) # worse case: each score takes one
    delta = 1#32
    unitw = min(max_weights)#1
    # one-hot
    if len(weights) == N:
        # weights start from 1
        # weights = [0] + weights
        # max_weights = [0] + max_weights
        weights.insert(0, 0)
        max_weights.insert(0, 0)

    selected_path = [[[0]*(N+1) for _ in range(M+1)] for _ in range(max_T)]
    update_path = [[-1]*(M+1) for _ in range(max_T)]

    best_T, final_idx = 0, -1
    last_visited = [[0]*(N+1) for _ in range(M+1)]
    last_selected = [[0]*(N+1) for _ in range(M+1)]
    last_growth = [[0]*(N+1) for _ in range(M+1)]
    last_dp = [0] * (M+1)
    for t in range(1, max_T):
        # NOTE: t-1 and t is continuous, so no need to check t-1,i,j, only t-1,N,: is needed for t
        visited = [[[0]*(N+1) for _ in range(M+1)] for _ in range(N+1)]
        selected = [[[0]*(N+1) for _ in range(M+1)] for _ in range(N+1)]
        alread_growth = [[[0]*(N+1) for _ in range(M+1)] for _ in range(N+1)]
        dp = [[0 for _ in range(M+1)] for _ in range(N+1)]
        update_from = [[-1 for _ in range(M+1)] for _ in range(N+1)]

        # natural growth of weights
        for j in range(0, M+1):
            # growth = sum(last_selected[j]) * unitw
            growth_list = [min(item+unitw, maxw)-item if flag else 0 
                                              for item, flag, maxw in zip(last_growth[j], last_selected[j], max_weights)]
            growth = sum(growth_list)
            if j+growth <= M and last_dp[j] > dp[0][j+growth]:
                dp[0][j+growth] = last_dp[j] + growth
                selected[0][j+growth] = [item for item in last_selected[j]]
                # alread_growth[0][j+growth] = [min(item+unitw, maxw) if flag else item 
                #                               for item, flag, maxw in zip(last_growth[j], last_selected[j], max_weights)]
                alread_growth[0][j+growth] = [item+gi for item, gi in zip(last_growth[j], growth_list)]
                visited[0][j+growth] = [item for item in last_visited[j]]
                update_from[0][j+growth] = j

        is_pop = [False] * (M+1)
        if t % delta == 0:
            # case 0: items pop
            cur_dp = [item for item in dp[0]]
            cur_selected = [[_item for _item in item] for item in selected[0]]
            cur_growth = [[_item for _item in item] for item in alread_growth[0]]
            cur_visited = [[_item for _item in item] for item in visited[0]]
            cur_from = [item for item in update_from[0]]
            for j in range(0, M+1):
                pop_idx = [0] * (N+1)
                _is_pop = False
                for k, item in enumerate(selected[0][j]):
                    assert max_weights[k] >= alread_growth[0][j][k]
                    if item and max_weights[k] == alread_growth[0][j][k]:
                        _is_pop = True
                        pop_idx[k] = 1
                if _is_pop:
                    released_capacity = sum([weights[idx] + alread_growth[0][j][idx] for idx, flag in enumerate(pop_idx) if flag])
                    if cur_dp[j-released_capacity] < dp[0][j]:
                        cur_dp[j-released_capacity] = dp[0][j]
                        cur_selected[j-released_capacity] = [0 if flag else item for item, flag in zip(selected[0][j], pop_idx)]
                        cur_growth[j-released_capacity] = [0 if flag else item for item, flag in zip(alread_growth[0][j], pop_idx)]
                        cur_visited[j-released_capacity] = [item for item in visited[0][j]]
                        is_pop[j-released_capacity] = True
                        cur_from[j-released_capacity] = update_from[0][j]
                        # update_from[t][i][j] = [t, i, j]
            dp[0] = cur_dp
            selected[0] = cur_selected
            alread_growth[0] = cur_growth
            visited[0] = cur_visited
            update_from[0] = cur_from

            # if sum(is_pop) > N//2:
            #     delta = delta // 2
            
        # normal 01 pack
        for i in range(1, N+1): # i also means the number of the current select items
            for j in range(0, M+1):
                # case 1: item i not selected
                case1 = dp[i-1][j]
                # case 2: item i selected
                _weight = weights[i]
                if j < _weight:
                    case2 = -1
                elif visited[i][j-_weight][i] or is_pop[j-_weight]:
                    case2 = -1
                else:
                    case2 = dp[i-1][j-_weight] + _weight
                # status transfer function
                mval, midx = best_of([case1, case2])
                if midx == 0:
                    # follow i-1,j
                    dp[i][j] = case1
                    visited[i][j] = [item for item in visited[i-1][j]]
                    selected[i][j] = [item for item in selected[i-1][j]]
                    alread_growth[i][j] = [item for item in alread_growth[i-1][j]]
                    update_from[i][j] = update_from[i-1][j]
                elif midx == 1:
                    # follow i-1,j-weights[i], and append i
                    dp[i][j] = case2
                    visited[i][j] = [item for item in visited[i-1][j-_weight]]
                    visited[i][j][i] = 1
                    selected[i][j] = [item for item in selected[i-1][j-_weight]]
                    selected[i][j][i] = 1
                    alread_growth[i][j] = [item for item in alread_growth[i-1][j-_weight]]
                    update_from[i][j] = update_from[i-1][j-_weight]
                # else: case3, do nothing

        # update last
        last_visited = visited[N]
        last_selected = selected[N]
        last_growth = alread_growth[N]
        last_dp = dp[N]

        # save necessary status
        selected_path[t] = selected[N]
        update_path[t] = update_from[N]

        if t % 128 == 0:
            # delta = min([64, delta*2])
            max_score = max(dp[N])
            max_idx = dp[N].index(max_score)
            print(f"t: {t}, max dp: {max_score}, final score: {final_score}," + 
                  f"visited: {sum(selected[N][max_idx])}, N: {N}, delta: {delta}, pop: {sum(is_pop)}")

        if max(dp[N]) >= final_score:
            best_T = t
            final_idx = dp[N].index(final_score)
            break
    # assert best_T > 0 and final_idx > -1, f"best_T: {best_T}, final_idx: {final_idx}"
    if best_T <= 0 or final_idx == -1:
        return -1, [dp[N]]

    # back track selected items for each T
    curj = final_idx
    selected_items = [[0]*(N+1) for _ in range(best_T+1)]
    for t in range(best_T, 0, -1):
        selected_items[t] = selected_path[t][curj]
        curj = update_path[t][curj]

    # restore item index to 0 start
    selected_items = [items[1:] for items in selected_items]

    return best_T, selected_items

@jit(nopython=True)
def knapsack(
        weights: List[int], max_weights: List[int], N: int, # N items, their weights, and max weight growth
        M: int, # capacity
    ) -> Tuple[int, List[List[int]]]: # T, and selected items for each T

    def best_of(values: List[int]) -> Tuple[int, int]:
        max_value = max(values)
        max_index = values.index(max_value)
        return max_value, max_index
    
    if N == 1:
        best_T = max_weights[0] + 1
        selected_items = [[1] for _ in range(best_T+1)]
        selected_items[0][0] = 0
        return best_T, selected_items
        
    original_weights = [item for item in weights]
    original_max_weights = [item for item in max_weights]
    original_M = M
    # normalization
    norm_term = max(min(weights), min(max_weights))
    weights = [math.ceil(item/norm_term) for item in weights]
    max_weights = [math.ceil(item/norm_term) for item in max_weights]
    M = math.ceil(M/norm_term)

    final_score = sum(weights) + sum(max_weights)
    max_T = final_score # worse case: each score takes one
    unitw = 1
    delta = 1
    # one-hot
    if len(weights) == N:
        # weights start from 1
        # weights = [0] + weights
        # max_weights = [0] + max_weights
        weights.insert(0, 0)
        max_weights.insert(0, 0)

    selected_path = [[[0]*(N+1) for _ in range(M+1)] for _ in range(max_T)]
    update_path = [[-1]*(M+1) for _ in range(max_T)]

    visited_path = [[[0]*(N+1) for _ in range(M+1)] for _ in range(max_T)]
    dp_path = [[0]*(M+1) for _ in range(max_T)]
    growth_path = [[[0]*(N+1) for _ in range(M+1)] for _ in range(max_T)]

    best_T, final_idx = 0, -1
    last_visited = [[0]*(N+1) for _ in range(M+1)]
    last_selected = [[0]*(N+1) for _ in range(M+1)]
    last_growth = [[0]*(N+1) for _ in range(M+1)]
    last_dp = [0] * (M+1)
    for t in range(1, max_T):
        # NOTE: t-1 and t is continuous, so no need to check t-1,i,j, only t-1,N,: is needed for t
        visited = [[[0]*(N+1) for _ in range(M+1)] for _ in range(N+1)]
        selected = [[[0]*(N+1) for _ in range(M+1)] for _ in range(N+1)]
        alread_growth = [[[0]*(N+1) for _ in range(M+1)] for _ in range(N+1)]
        dp = [[0 for _ in range(M+1)] for _ in range(N+1)]
        update_from = [[-1 for _ in range(M+1)] for _ in range(N+1)]

        # natural growth of weights
        for j in range(0, M+1):
            # growth = sum(last_selected[j]) * unitw
            growth_list = [min(item+unitw, maxw)-item if flag else 0 
                                              for item, flag, maxw in zip(last_growth[j], last_selected[j], max_weights)]
            growth = sum(growth_list)
            if j+growth <= M and last_dp[j] > dp[0][j+growth]:
                dp[0][j+growth] = last_dp[j] + growth
                selected[0][j+growth] = [item for item in last_selected[j]]
                # alread_growth[0][j+growth] = [min(item+unitw, maxw) if flag else item 
                #                               for item, flag, maxw in zip(last_growth[j], last_selected[j], max_weights)]
                alread_growth[0][j+growth] = [item+gi for item, gi in zip(last_growth[j], growth_list)]
                visited[0][j+growth] = [item for item in last_visited[j]]
                update_from[0][j+growth] = j

        is_pop = [False] * (M+1)
        # case 0: items pop
        if t % delta == 0:
            cur_dp = [item for item in dp[0]]
            cur_selected = [[_item for _item in item] for item in selected[0]]
            cur_growth = [[_item for _item in item] for item in alread_growth[0]]
            cur_visited = [[_item for _item in item] for item in visited[0]]
            cur_from = [item for item in update_from[0]]
            for j in range(0, M+1):
                pop_idx = [0] * (N+1)
                _is_pop = False
                for k, item in enumerate(selected[0][j]):
                    assert max_weights[k] >= alread_growth[0][j][k]
                    if item and max_weights[k] == alread_growth[0][j][k]:
                        _is_pop = True
                        pop_idx[k] = 1
                if _is_pop:
                    released_capacity = sum([weights[idx] + alread_growth[0][j][idx] for idx, flag in enumerate(pop_idx) if flag])
                    if cur_dp[j-released_capacity] < dp[0][j]:
                        cur_dp[j-released_capacity] = dp[0][j]
                        cur_selected[j-released_capacity] = [0 if flag else item for item, flag in zip(selected[0][j], pop_idx)]
                        cur_growth[j-released_capacity] = [0 if flag else item for item, flag in zip(alread_growth[0][j], pop_idx)]
                        cur_visited[j-released_capacity] = [item for item in visited[0][j]]
                        is_pop[j-released_capacity] = True
                        cur_from[j-released_capacity] = update_from[0][j]
                        # update_from[t][i][j] = [t, i, j]
                    dp[0][j] = 0
                    selected[0][j] = [0] * (N+1)
                    alread_growth[0][j] = [0] * (N+1)
                    visited[0][j] = [0] * (N+1)
                    update_from[0][j] = -1
            dp[0] = cur_dp
            selected[0] = cur_selected
            alread_growth[0] = cur_growth
            visited[0] = cur_visited
            update_from[0] = cur_from

            # if sum(is_pop) > N//2:
            #     delta = 1
            
        # normal 01 pack
        for i in range(1, N+1): # i also means the number of the current select items
            for j in range(0, M+1):
                # case 1: item i not selected
                case1 = dp[i-1][j]
                # case 2: item i selected
                _weight = weights[i]
                if j < _weight:
                    case2 = -1
                elif visited[i][j-_weight][i]:# or is_pop[j-_weight]:
                    case2 = -1
                else:
                    case2 = dp[i-1][j-_weight] + _weight
                # status transfer function
                mval, midx = best_of([case1, case2])
                if midx == 0:
                    # follow i-1,j
                    dp[i][j] = case1
                    visited[i][j] = [item for item in visited[i-1][j]]
                    selected[i][j] = [item for item in selected[i-1][j]]
                    alread_growth[i][j] = [item for item in alread_growth[i-1][j]]
                    update_from[i][j] = update_from[i-1][j]
                elif midx == 1:
                    # follow i-1,j-weights[i], and append i
                    dp[i][j] = case2
                    visited[i][j] = [item for item in visited[i-1][j-_weight]]
                    visited[i][j][i] = 1
                    selected[i][j] = [item for item in selected[i-1][j-_weight]]
                    selected[i][j][i] = 1
                    alread_growth[i][j] = [item for item in alread_growth[i-1][j-_weight]]
                    update_from[i][j] = update_from[i-1][j-_weight]
                # else: case3, do nothing

        # update last
        last_visited = visited[N]
        last_selected = selected[N]
        last_growth = alread_growth[N]
        last_dp = dp[N]

        # save necessary status
        selected_path[t] = selected[N]
        update_path[t] = update_from[N]

        visited_path[t] = visited[N]
        dp_path[t] = dp[N]
        growth_path[t] = alread_growth[N]

        if t % 128 == 0:
            # delta = min(64, delta*2)
            max_score = max(dp[N])
            max_idx = dp[N].index(max_score)
            print(f"t: {t}, max dp: {max_score}, final score: {final_score}," + 
                  f"visited: {sum(selected[N][max_idx])}, N: {N}, delta: {delta}, pop: {sum(is_pop)}")

        if max(dp[N]) >= final_score:
            best_T = t
            final_idx = dp[N].index(max(dp[N]))
            print(f"max dp: {max(dp[N])}")
            break

    # assert best_T > 0 and final_idx > -1, f"best_T: {best_T}, final_idx: {final_idx}"
    if best_T <= 0 or final_idx == -1:
        best_T = t
        max_dp = max(dp[N])
        final_idx = dp[N].index(max_dp)
        final_selected = selected[N][final_idx]
        num_repeat = max([mi-gi for si, gi, mi in zip(
            final_selected, alread_growth[N][final_idx], max_weights) if si])
        # back track selected items for each T
        curj = final_idx
        selected_items = [[0]*(N+1) for _ in range(best_T+1+num_repeat)]
        for t in range(best_T, 1, -1):
            selected_items[t] = selected_path[t][curj]
            curj = update_path[t][curj]
        for t in range(best_T+1, best_T+1+num_repeat):
            selected_items[t] = final_selected

        # restore item index to 0 start
        selected_items = [items[1:] for items in selected_items for _ in range(norm_term-1) if sum(items)>0]
        selected_items.insert(0, [0]*(N))

        print(f"Exceeding max_T: max dp: {max_dp}, final score: {final_score}")

        return len(selected_items), selected_items
        # return -1, [dp[N]]

    # back track selected items for each T
    curj = final_idx
    selected_items = [[0]*(N+1) for _ in range(best_T+1)]
    for t in range(best_T, 0, -1):
        selected_items[t] = selected_path[t][curj]
        curj = update_path[t][curj]

    # restore item index to 0 start
    selected_items = [items[1:] for items in selected_items for _ in range(norm_term-1) if sum(items)>0]
    selected_items.insert(0, [0]*(N))

    return len(selected_items), selected_items

class SchedulerV3(SchedulerV2):
    def __init__(
        self, 
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig]
    ) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)

        self.schedule_plan = deque()
        self.best_t = 0
        self.idx_to_seq: dict[int, SequenceGroup] = {}
        self.extra_waiting = deque()
        self.unscheduled = set()

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)
        self.extra_waiting.append(seq_group)
  
    def pre_schedule(self, block_ratio=0.025, max_seq=100) -> None:
        # NOTE: ensure remaining_decode is updated before pre_schedule
        def token_to_block(num_token: int) -> int:
            return num_token // self.block_manager.block_size + 1
        def compute_weights(seq_group: SequenceGroup) -> int:
            num_prompt = len(seq_group.get_seqs()[0].get_prompt_token_ids())
            num_decode = seq_group.remaining_decode
            num_total = num_prompt+num_decode
            num_extra = token_to_block(num_total) * self.block_manager.block_size - num_total
            assert num_extra >= 0 and num_decode>0
            return num_prompt, num_decode, num_extra

        # # sort
        # now = time.time()
        # policy = PolicyFactory.get_policy(policy_name="sdf")
        # waiting = policy.sort_by_priority(now, self.waiting)

        waiting = list(self.waiting)
        all_items = waiting[:max_seq]
        self.extra_waiting = deque(waiting[max_seq:])
        # assign remaining_decode
        for item in all_items:
            if item.is_prefill() and item.remaining_decode == 0:
                self.length_predictor.assign_one(item)
                item.just_end = 0
        results = [compute_weights(item) for item in all_items]
        weights = numba.typed.List([item[0] for item in results])
        max_weights = numba.typed.List([item[1] for item in results])
        extra = sum([item[2] for item in results])
        N = len(all_items)
        # M = int(block_ratio * self.block_manager.get_num_free_gpu_blocks()) * self.block_manager.block_size - extra
        # M = max(weights)+max(max_weights)
        M = sum(weights) + max(max_weights)
        # assert max(weights)+max(max_weights) <= M, f"{max(weights), max(max_weights), M}"
        t0 = time.time()
        best_t, schedule_plan = knapsack(
            weights, max_weights, N, M)
        # assert best_t != -1, "knapsack failed"
        assert len(schedule_plan[-1]) == N
        logger.info(f"knapsack takes {time.time()-t0: .2f}s")
        max_idx = max(self.idx_to_seq.keys() or [-1]) + 1
        for t in range(best_t):
            schedule_plan[t] = [i+max_idx for i, flag in enumerate(schedule_plan[t]) if flag]
        idx_to_seq = {
            i+max_idx: item for i, item in enumerate(all_items)}
        # NOTE: compatible with new requests coming
        self.schedule_plan += schedule_plan[:best_t] # TODO: sometimes best_t!=len(schedule_plan)
        self.unscheduled = set(chain(*self.schedule_plan))
        assert len(self.unscheduled) == N, f"{len(self.unscheduled), N}"
        self.best_t = max_idx
        # self.idx_to_seq = {key: self.idx_to_seq[key] for key in self.unscheduled}
        self.idx_to_seq = {key: val for key, val in self.idx_to_seq.items() if key>=self.best_t}
        self.idx_to_seq.update(idx_to_seq)

        logger.info(f"{sum([len(item)==1 for item in self.schedule_plan])/len(self.schedule_plan)*100: .2f}% one decoding.")

    def get_one_schedule(self) -> Tuple[deque, deque, deque, deque]:
        if len(self.schedule_plan) == 0:
            if len(self.waiting) == 0:
                return deque(), deque(), deque(), deque()
            logger.info("Run pre-schedule.")
            self.pre_schedule()
            self.unfinished = set()

        total_seq_groups = len(self.waiting) + len(self.running) + len(self.swapped)
        # NOTE: we do not have swap in, only RECOMPUTE preempt is used
        last_schedule = set(self.schedule_plan[0])
        next_schedule = set(self.schedule_plan[1]) if len(self.schedule_plan) > 1 else set()
        prefilling = next_schedule - last_schedule
        swapped_out = last_schedule - next_schedule
        self.unfinished = self.unfinished | swapped_out
        running = next_schedule & last_schedule
        self.unscheduled -= next_schedule
        waiting = self.unscheduled | self.unfinished

        assert len(self.idx_to_seq) <= len(waiting) + len(running) + len(prefilling), \
            f"{len(self.idx_to_seq), len(waiting), len(running), len(prefilling)}"

        prefilling = deque([self.idx_to_seq[i] for i in prefilling 
                            if i in self.idx_to_seq])
        swapped_out = deque([self.idx_to_seq[i] for i in swapped_out 
                             if i in self.idx_to_seq])
        running = deque([self.idx_to_seq[i] for i in running 
                         if i in self.idx_to_seq])
        waiting = deque([self.idx_to_seq[i] for i in waiting 
                         if i in self.idx_to_seq])
        assert total_seq_groups == len(waiting) + len(running) + len(prefilling) + len(self.extra_waiting), \
            f"{total_seq_groups, len(waiting), len(running), len(prefilling), len(self.extra_waiting)}"

        self.schedule_plan.popleft()
        # NOTE: waiting is put last, they have been served
        return running, swapped_out, prefilling, self.extra_waiting + waiting
    
    def free_finished_seq_groups(self) -> None:
        length = len(self.idx_to_seq)
        # update schedule_plan
        self.idx_to_seq = {key: seq_group for key, seq_group in self.idx_to_seq.items()
                           if not seq_group.is_finished()}
        if length-len(self.idx_to_seq) > 0:
            logger.info(f"free {length-len(self.idx_to_seq)} finished seq_groups")

        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())

    def _schedule_chunked_prefill_with_predicted_length(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        total_seq_groups = len(self.waiting) + len(self.running) + len(self.swapped)
        budget = SchedulingBudget(
            token_budget=10240, # 512 by default
            max_num_seqs=1024, # 256 by default
        )

        running, swapped_out, prefilling, waiting = self.get_one_schedule()

        # swapped_out: preempt and put into waiting
        _, decode_preempted = self._update_running_decode(swapped_out)
        
        # running: update remaining decode and then scheduled
        self._update_remaining_decode(running)
        _, running_scheduled = self._schedule_running(
            running, budget, enable_chunking=True)

        # prefilling: scheduled prefill and put into running
        _, prefills = self._schedule_prefills(
            prefilling, budget, enable_chunking=True)


        # assert (budget.num_batched_tokens <=
        #         self.scheduler_config.max_num_batched_tokens)
        # assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        swapped_in = SchedulerSwappedInOutputs.create_empty()
        # Update waiting requests.
        self.waiting = waiting
        # Update new running requests.
        self.running = running + prefilling
        # update metrics
        assert total_seq_groups == len(self.waiting) + len(self.running) + len(self.swapped) + len(prefills.ignored_seq_groups)
        self._update_time_metrcis(prefills, running_scheduled, 
                                swapped_in, decode_preempted)

        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + 
                            decode_preempted.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
                        swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) + 
                    len(decode_preempted.preempted)),
        )

    def _update_running_decode(
        self,
        running_queue: deque,
        budget: Optional[SchedulingBudget]=None,
        curr_loras: Optional[Set[int]]=None,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        '''
        preempt all running seq_group
        '''
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []
        remaining_running: Deque[Sequence] = deque()

        for seq_group in running_queue:
            # NOTE: use SWAP because RECOMPUTE loses remaining decode
            # but remaining_decode=0, so finally use RECOMPUTE
            self._preempt(seq_group,
                        blocks_to_swap_out,
                        PreemptionMode.RECOMPUTE)
            preempted.append(seq_group)

        return remaining_running, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=0)
    
    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]]=None,
        policy: Optional[Policy]=None,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        now = time.time()

        for seq_group in running_queue:
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)
            assert self._can_append_slots(seq_group) and num_running_tokens != 0
            budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
            num_running_seqs = seq_group.get_max_num_running_seqs()
            budget.subtract_num_seqs(seq_group.request_id,
                                        num_running_seqs)
            
            t0 = time.time()
            self._append_slots(seq_group, blocks_to_copy)
            self.cache_consumption += time.time() - t0
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(
                        seq_group=seq_group,
                        token_chunk_size=num_running_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group=seq_group,
                                            token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id,
                                            num_running_tokens)
            # OPTIMIZATION:  Note that get_max_num_running_seqs is
            # expensive. For the default scheduling chase where
            # enable_chunking is False, num_seqs are updated before running
            # this method, so we don't have to update it again here.
            if enable_chunking:
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.add_num_seqs(seq_group.request_id, num_running_seqs)

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=0)
    
    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]]=None,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []

        for seq_group in waiting_queue:
            self._passed_delay(time.time())
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            assert can_allocate != AllocStatus.LATER
            if can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            assert num_new_tokens != 0 and budget.can_schedule(
                num_new_tokens=num_new_tokens, num_new_seqs=num_new_seqs)

            t0 = time.time()
            self._allocate_and_set_running(seq_group)
            self.cache_consumption += time.time() - t0

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=0)
    

class descriptor:
    def __set_name__(self, owner, name):
        self.name = f'_{name}'
    def __get__(self, obj, objtype=None):
        return obj.extra_dict.get(self.name, None)
    def __set__(self, obj, value):
        obj.extra_dict[self.name] = value

@dataclass
class ScheduledSequenceGroupV2(ScheduledSequenceGroup):
    do_sample: Optional[bool]=True

class LazySequenceGroup:
    # num_new_tokens = descriptor()
    # lazy_state = descriptor()
    # new_seq_states = descriptor()
    # new_data_stages = descriptor()
    # new_data_num_computed_tokens = descriptor()
    # new_data_len = descriptor()
    __slots__ = ('seq_group', 'num_new_tokens', 
                    'lazy_state', 'new_seq_states',
                    'new_data_stages', 'new_data_num_computed_tokens',
                    'new_data_len', 'history', 'do_sample',)
    def __init__(
        self, 
        seq_group: SequenceGroup, 
        num_new_tokens: Optional[int]=0,
        new_seq_states: Optional[List[SequenceStatus]]=None,
        new_data_stages: Optional[List[SequenceStage]]=None,
        new_data_num_computed_tokens: Optional[List[int]]=None,
        new_data_len: Optional[List[int]]=None,
        lazy_state: Optional[LazyStates]=None,
        history: Optional[deque]=None,
        do_sample: Optional[bool]=True,
    ) -> None:
        # super().__init__(
        #     request_id=seq_group.request_id, 
        #     seqs=list(seq_group.seqs_dict.values()), 
        #     arrival_time=seq_group.metrics.arrival_time)
        # self.__dict__ = seq_group.__dict__ # this means we never separate lazy and non-lazy
        assert len(seq_group.get_seqs()) == 1
        # self.extra_dict = {}

        self.seq_group = seq_group

        self.num_new_tokens = num_new_tokens
        self.lazy_state = lazy_state
        self.do_sample = do_sample

        if history is None:
            self.history = deque()
        else:
            self.history = history

        if new_seq_states is None:
            self.new_seq_states = [seq_group.get_seqs()[0].status]
        else:
            self.new_seq_states = new_seq_states

        if new_data_stages is None:
            self.new_data_stages = [seq_group.get_seqs()[0].data.stage]
        else:
            self.new_data_stages = new_data_stages

        if new_data_num_computed_tokens is None:
            self.new_data_num_computed_tokens = \
                [seq_group.get_seqs()[0].data.get_num_computed_tokens()]
        else:
            self.new_data_num_computed_tokens = new_data_num_computed_tokens

        if new_data_len is None:
            self.new_data_len = [seq_group.get_seqs()[0].data.get_len()]
        else:
            self.new_data_len = new_data_len

    # @property
    # def num_new_tokens(self) -> int:
    #     return self.extra_dict.get("num_new_tokens", 0)

    # @num_new_tokens.setter
    # def num_new_tokens(self, num_new_tokens: int) -> None:
    #     self.extra_dict["num_new_tokens"] = num_new_tokens

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped object (B).
        """
        if name in self.__slots__:
            return getattr(self, name)
        else:
            return getattr(self.seq_group, name)#getattr(self.seq_group, name)

    def __setattr__(self, name: str, value: enum.Any) -> None:
        if name in self.__slots__:
            super().__setattr__(name, value)#setattr(self, name, value)
        else:
            setattr(self.seq_group, name, value)#setattr(self.seq_group, name, value)

    # def update(self, **kwargs) -> LazySequenceGroup:
    #     self.__dict__.update(kwargs)
    #     return self

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        return list(self.seqs_dict.values()) if status is None else [
            seq for seq, seq_states in zip(self.seqs_dict.values(), self.new_seq_states) if seq_states == status
        ]

    def _get_num_new_tokens(
        self,
        status: SequenceStatus,
        enable_chunking: bool,
        budget: SchedulingBudget
    ) -> int:
        num_new_tokens = 0
        assert self.new_seq_states[0] == status
        if self.new_data_stages[0] == SequenceStage.DECODE:
            num_new_tokens = 1
        else:
            num_new_tokens = (self.new_data_len[0] - self.new_data_num_computed_tokens[0])
        assert num_new_tokens > 0
        # Chunk if a running request cannot fit in.
        # If number of seq > 1, it means it is doing beam search in a
        # decode phase. Do not chunk in that case.
        if enable_chunking:
            num_new_tokens = min(num_new_tokens,
                                 budget.remaining_token_budget())
        return num_new_tokens

    # def get_unfinished_seqs(self) -> List[Sequence]:
    #     if SequenceStatus.is_finished(self.new_seq_states[0]):
    #         return []
    #     else:
    #         return list(self.seqs_dict.values())

    # def get_finished_seqs(self) -> List[Sequence]:
    #     if not SequenceStatus.is_finished(self.new_seq_states[0]):
    #         return []
    #     else:
    #         return list(self.seqs_dict.values())
    
    # def is_finished(self) -> bool:
    #     return SequenceStatus.is_finished(self.new_seq_states[0])

    def is_prefill(self) -> bool:
        # return self.new_data_len[0] > self.new_data_num_computed_tokens[0]
        return self.new_data_stages[0] == SequenceStage.PREFILL
    
    def _preempt_by_recompute(self) -> None:
        assert self.new_seq_states[0] == SequenceStatus.RUNNING
        self.new_seq_states[0] = SequenceStatus.WAITING
        self.new_data_num_computed_tokens[0] = 0
        self.new_data_stages[0] = SequenceStage.PREFILL

    def _preempt_by_swap(self) -> None:
        if self.new_seq_states == SequenceStatus.RUNNING:
            self.new_seq_states[0] = SequenceStatus.SWAPPED

    def update_by_self(self) -> None:
        seqs = self.get_seqs()[0]
        self.num_new_tokens = 0
        self.lazy_state = None
        self.new_seq_states = [seqs.status]
        self.new_data_stages = [seqs.data.stage]
        self.new_data_num_computed_tokens = \
                [seqs.data.get_num_computed_tokens()]
        self.new_data_len = [seqs.data.get_len()]
        self.history = deque()
        self.do_sample = True

class LazyScheduler(SchedulerV2):
    def __init__(self, 
                scheduler_config: SchedulerConfig,
                cache_config: CacheConfig,
                lora_config: Optional[LoRAConfig],
                queue_size: Optional[int] = 1) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        self.free_gpu_blocks = None# self.block_manager.gpu_allocator.get_num_free_blocks()
        self.gpu_consume_records = []
        # pipeline
        self.recorder_queue: Queue[LazySchedule, LazySchedule, LazySchedule, SchedulingBudget] = Queue(maxsize=queue_size)
        self.scheduling_count = 0
        self.execution_count  = 0
        self.token_queue: Queue[str] = Queue(maxsize=1)
        self.token_queue.put("exceeding_1")

        self.atomic = threading.Lock()
        self.is_sync = False
        self.sync_event = threading.Event()

    def _schedule(self) -> SchedulerOutputs:
        # return self._schedule_chunked_prefill_with_predicted_length()
        # return self._test_schedule_chunked_prefill_with_predicted_length()
        return self._lazy_executor()

    def _test_schedule_chunked_prefill_with_predicted_length(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        total_seq_groups = len(self.waiting) + len(self.running) + len(self.swapped)

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                    SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        # self.running, decode_preempted = self._update_running_decode(
        #     self.running,
        #     budget,
        #     curr_loras,
        #     enable_chunking=True)
        self._update_remaining_decode(self.running)
        decode_preempted = SchedulerRunningOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        # fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        df_policy = PolicyFactory.get_policy(policy_name="sdf")

        # test lazy
        self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()

        # fork_budget = copy.deepcopy(budget)
        # fork_running = copy.deepcopy(self.running)
        # estimated_remaing_running, estimated_running_scheduled = self._estimate_schedule_running(
        #     fork_running,
        #     fork_budget,
        #     curr_loras,
        #     df_policy,
        #     enable_chunking=True)

        remaining_running, running_scheduled = self._schedule_running(
            self.running,
            budget,
            curr_loras,
            df_policy,
            enable_chunking=True)

        # assert set([str(s.seq_group) for s in running_scheduled.decode_seq_groups]) == \
        #     set([str(s.seq_group) for s in estimated_running_scheduled if s.state == LazyStates.DECODE])
        # assert set([str(s.seq_group) for s in running_scheduled.prefill_seq_groups]) == \
        #     set([str(s.seq_group) for s in estimated_running_scheduled if s.state == LazyStates.PREFILL])
        # assert set([str(s) for s in running_scheduled.preempted]) == \
        #     set([str(s.seq_group) for s in estimated_running_scheduled if s.state == LazyStates.RECOMPUTE])
        # assert set([str(s) for s in running_scheduled.swapped_out]) == \
        #     set([str(s.seq_group) for s in estimated_running_scheduled if s.state == LazyStates.SWAP])
        # assert set([str(s) for s in estimated_remaing_running]) == set([str(s) for s in remaining_running])
        # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) + len(
                    decode_preempted.swapped_out) + len(
                        decode_preempted.preempted) == 0:
            # self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
            # fork_budget = copy.deepcopy(budget)
            # fork_swapped = copy.deepcopy(self.swapped)
            # estimated_remaining_swapped, estimated_swapped_in = self._estimate_schedule_swapped(
            #     fork_swapped,
            #     fork_budget,
            #     curr_loras,
            #     df_policy)

            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, df_policy)
            
            # assert set([str(s) for s in remaining_swapped]) == set([str(s) for s in estimated_remaining_swapped])
            # assert set([str(s.seq_group) for s in swapped_in.decode_seq_groups]) == \
            #     set([str(s.seq_group) for s in estimated_swapped_in if s.state == LazyStates.DECODE])
            # assert set([str(s.seq_group) for s in swapped_in.prefill_seq_groups]) == \
            #     set([str(s.seq_group) for s in estimated_swapped_in if s.state == LazyStates.PREFILL])
            # assert set([str(s) for s in swapped_in.infeasible_seq_groups]) == \
            #     set([str(s.seq_group) for s in estimated_swapped_in if s.state == LazyStates.NEVER])
            # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()
            

        # Schedule new prefills.
        # self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
        # fork_budget = copy.deepcopy(budget)
        # fork_waiting = copy.deepcopy(self.waiting)
        # estimated_remaining_waiting, estimated_prefills = self._estimate_schedule_prefills(
        #     fork_waiting,
        #     fork_budget,
        #     curr_loras,
        #     enable_chunking=True)

        remaining_waiting, prefills = self._schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)
        
        # assert set([str(s) for s in estimated_remaining_waiting]) == set([str(s) for s in remaining_waiting])
        # assert set([str(s.seq_group) for s in prefills.seq_groups]) == \
        #     set([str(s.seq_group) for s in estimated_prefills if s.state == LazyStates.PREFILL])
        # assert set([str(s) for s in prefills.ignored_seq_groups]) == \
        #     set([str(s.seq_group) for s in estimated_prefills if s.state == LazyStates.IGNORE])
        # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        self.waiting.extendleft(decode_preempted.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        self.swapped.extend(decode_preempted.swapped_out)
        # update metrics
        assert total_seq_groups == len(self.waiting) + len(self.running) + len(self.swapped)
        self._update_time_metrcis(prefills, running_scheduled, 
                                swapped_in, decode_preempted)

        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + 
                            decode_preempted.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
                        swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) + 
                    len(decode_preempted.preempted)),
        )

    def _maybe_can_append_slots(self,
                         seq_group: LazySequenceGroup,
                         num_lookahead_slots: Optional[int] = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerV1"

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= self.free_gpu_blocks

    def _lazy_preempt(
        self,
        seq_group: LazySequenceGroup,
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        if preemption_mode is None:
            if seq_group.remaining_decode >= 1:
                # prioritize remaining decode
                preemption_mode = PreemptionMode.SWAP
            elif seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        # physical_block = self.block_manager._get_physical_blocks(seq_group)
        # self.free_gpu_blocks += len(physical_block)

        if preemption_mode == PreemptionMode.RECOMPUTE:
            physical_block = self.block_manager._get_physical_blocks(seq_group)
            self.free_gpu_blocks += len(physical_block)
            seq_group._preempt_by_recompute()
        else:
            if not self.block_manager.can_swap_out(seq_group):
                # FIXME(woosuk): Abort the sequence group instead of aborting the
                # entire engine.
                raise RuntimeError(
                    "Aborted due to the lack of CPU swap space. Please increase "
                    "the swap space to avoid this error.")
            physical_block = self.block_manager._get_physical_blocks(seq_group)
            self.free_gpu_blocks += len(physical_block)
            seq_group._preempt_by_swap()

        return preemption_mode

    def _lazy_append_slots(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            len_logical_blocks = len(seq.logical_token_blocks)
            if seq.seq_id in self.block_manager.block_tables:
                len_block_table = len(self.block_manager.block_tables[seq.seq_id])
            else:
                len_block_table = 0
            if len_block_table < len_logical_blocks:
                # Currently this code only supports adding one physical block
                assert len_block_table == len_logical_blocks - 1
                self.free_gpu_blocks -= 1

    def _estimate_schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, deque]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        assert not self.lora_enabled, 'curr_loras is not supported in lazy scheduling'
        # Record operations that are executed later.
        recorder: Deque[LazySequenceGroup] = deque()

        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)
        # running_queue = self.running

        while running_queue:
            seq_group: LazySequenceGroup = running_queue[0]
            # num_running_tokens = self._get_num_new_tokens(
            #     seq_group, SequenceStatus.RUNNING, enable_chunking, budget)
            num_running_tokens = seq_group._get_num_new_tokens(
                SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            while not self._maybe_can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                t0 = time.time()
                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()
                    preempted_mode = self._lazy_preempt(victim_seq_group)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        victim_seq_group.lazy_state = LazyStates.RECOMPUTE
                        victim_seq_group.num_new_tokens = 0
                        recorder.append(victim_seq_group)
                    else:
                        victim_seq_group.lazy_state = LazyStates.SWAP
                        victim_seq_group.num_new_tokens = 0
                        recorder.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._lazy_preempt(seq_group)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        seq_group.lazy_state = LazyStates.RECOMPUTE
                        seq_group.num_new_tokens = 0
                        recorder.append(seq_group)
                    else:
                        seq_group.lazy_state = LazyStates.SWAP
                        seq_group.num_new_tokens = 0
                        recorder.append(seq_group)
                    break
            else:
                self._lazy_append_slots(seq_group)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    seq_group.lazy_state = LazyStates.PREFILL
                    seq_group.num_new_tokens = num_running_tokens
                    recorder.append(seq_group)
                else:
                    seq_group.lazy_state = LazyStates.DECODE
                    seq_group.num_new_tokens = 1
                    recorder.append(seq_group)
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)

        return running_queue, recorder

    def _maybe_can_swap_in(
        self,
        seq_group: LazySequenceGroup,
        num_lookahead_slots: int = 0
    ) -> AllocStatus:
        assert (num_lookahead_slots == 0
                ), "BlockSpaceManagerV1 does not support lookahead allocation"

        blocks = self.block_manager._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        if seq_group.is_encoder_decoder():
            num_swapped_seqs += 1
        num_free_blocks = self.free_gpu_blocks
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        if self.block_manager.gpu_allocator.get_num_total_blocks() < num_required_blocks:
            return AllocStatus.NEVER
        elif num_free_blocks - num_required_blocks >= self.block_manager.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _lazy_swap_in(
        self,
        seq_group: LazySequenceGroup,
    ) -> None:
        if seq_group.new_seq_states[0] == SequenceStatus.SWAPPED:
            num_logical_blocks = len(seq_group.get_seqs()[0].logical_token_blocks)
            self.free_gpu_blocks -= num_logical_blocks
            seq_group.new_seq_states[0] = SequenceStatus.RUNNING

    def _estimate_schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, deque]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        assert not self.lora_enabled, 'curr_loras is not supported in lazy scheduling'
        # record operations that are executed later.
        recorder: Deque[LazySequenceGroup] = deque()

        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)

        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            alloc_status = self._maybe_can_swap_in(seq_group)
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                # for seq in seq_group.get_seqs():
                #     seq.status = SequenceStatus.FINISHED_IGNORED
                seq_group.new_seq_states[0] = SequenceStatus.FINISHED_IGNORED
                recorder.append(seq_group)
                swapped_queue.popleft()
                continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = seq_group._get_num_new_tokens(SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            swapped_queue.popleft()

            t0 = time.time()
            self._lazy_swap_in(seq_group)
            self._lazy_append_slots(seq_group)
            self.cache_consumption += time.time() - t0

            is_prefill = seq_group.is_prefill()
            if is_prefill:
                seq_group.lazy_state = LazyStates.PREFILL
                seq_group.num_new_tokens = num_new_tokens
                recorder.append(seq_group)
            else:
                seq_group.lazy_state = LazyStates.DECODE
                seq_group.num_new_tokens = 1
                recorder.append(seq_group)
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        return swapped_queue, recorder

    def _maybe_can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self.block_manager, seq_group)

        self_num_required_blocks = self.block_manager._get_seq_num_required_blocks(
            seq_group.get_seqs(status=SequenceStatus.WAITING)[0])
        cross_num_required_blocks = self.block_manager._get_seq_num_required_blocks(
            seq_group.get_encoder_seq())
        num_required_blocks = self_num_required_blocks + \
                              cross_num_required_blocks

        if self.block_manager.block_sliding_window is not None:

            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.free_gpu_blocks

        # Use watermark to avoid frequent cache eviction.
        if (self.block_manager.num_total_gpu_blocks - num_required_blocks <
                self.block_manager.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.block_manager.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _lazy_allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        is_encoder_decoder = seq_group.is_encoder_decoder()
        check_no_caching_or_swa_for_blockmgr_encdec(self.block_manager, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_prompt_blocks = len(seq.logical_token_blocks)
        self.free_gpu_blocks -= num_prompt_blocks

        if is_encoder_decoder:
            seq_ = seq_group.get_encoder_seq()
            num_prompt_blocks_ = len(seq_.logical_token_blocks)
            self.free_gpu_blocks -= num_prompt_blocks_

        if seq_group.new_seq_states[0] == SequenceStatus.WAITING:
            seq_group.new_seq_states[0] = SequenceStatus.RUNNING
        # for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
        #     seq.status = SequenceStatus.RUNNING

    def _estimate_schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, deque]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        assert not self.lora_enabled, 'curr_loras is not supported in lazy scheduling'
        waiting_queue = deque([s for s in waiting_queue])
        # record operations that are executed later.
        recorder: Deque[LazySequenceGroup] = deque()

        leftover_waiting_sequences: Deque[LazySequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = seq_group._get_num_new_tokens(SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                # for seq in waiting_seqs:
                #     seq.status = SequenceStatus.FINISHED_IGNORED
                seq_group.new_seq_states[0] = SequenceStatus.FINISHED_IGNORED
                # ignored_seq_groups.append(seq_group)
                seq_group.lazy_state = LazyStates.IGNORE
                seq_group.num_new_tokens = 0
                recorder.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            # can_allocate = self.block_manager.can_allocate(seq_group)
            can_allocate = self._maybe_can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                # for seq in waiting_seqs:
                #     seq.status = SequenceStatus.FINISHED_IGNORED
                seq_group.new_seq_states[0] = SequenceStatus.FINISHED_IGNORED
                # ignored_seq_groups.append(seq_group)
                seq_group.lazy_state = LazyStates.IGNORE
                seq_group.num_new_tokens = 0
                recorder.append(seq_group)
                waiting_queue.popleft()
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            waiting_queue.popleft()

            t0 = time.time()
            self._lazy_allocate_and_set_running(seq_group)
            self.cache_consumption += time.time() - t0

            seq_group.lazy_state = LazyStates.PREFILL
            assert num_new_tokens > 0
            seq_group.num_new_tokens = num_new_tokens
            recorder.append(seq_group)
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len([r for r in recorder if r.lazy_state == LazyStates.PREFILL]) > 0:
            self.prev_prompt = True

        return waiting_queue, recorder
    
    def lazy_schedule(self) -> float:
        if self.free_gpu_blocks is None:
            self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
        return self._lazy_schedule_chunked_prefill_with_predicted_length()
    
    def _schedule(self) -> SchedulerOutputs:
        return self._lazy_executor()

    def _lazy_schedule_chunked_prefill_with_predicted_length(
        self
    ) -> float:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        if self.get_num_running_requests() == 0:
            logger.warning("No request running, scheduler return.")
            return 0

        t_0 = time.time()
        try:
            token = self.token_queue.get(timeout=10)#(timeout=1)
        except Exception as e:
            logger.warning(f"Lazy scheduler timeout for acquiring token")
            return 0
        t0 = time.time()
        self.scheduling_count += 1
        
        # sync processing
        if self.is_sync:
            self.cache_lock()
            logger.warning("Sync found, call sync_running.")
            self.schedule_restore()
            self.is_sync = False
            self.cache_unlock()
            self.sync_event.set()

        self.atomic.acquire()
        try:
            logger.info("Scheduler is running.")
            budget = SchedulingBudget(
                token_budget=self.scheduler_config.max_num_batched_tokens,
                max_num_seqs=self.scheduler_config.max_num_seqs,
            )
            curr_loras: Set[int] = set()

            remaining_waiting = deque([
                LazySequenceGroup(seq_group) if type(seq_group) != LazySequenceGroup else seq_group
                for seq_group in self.waiting])
            remaining_running = deque([
                LazySequenceGroup(seq_group) if type(seq_group) != LazySequenceGroup else seq_group
                for seq_group in self.running])
            remaining_swapped = deque([
                LazySequenceGroup(seq_group) if type(seq_group) != LazySequenceGroup else seq_group 
                for seq_group in self.swapped])
            # assert len(remaining_running) == len(self.running)
            # assert len(remaining_swapped) == len(self.swapped)
            # assert len(remaining_waiting) == len(self.waiting)
            total_len = len(self.running) + len(self.waiting) + len(self.swapped)

            self._update_remaining_decode(remaining_running)
            decode_preempted = SchedulerRunningOutputs.create_empty()

            gpu_consume = [self.free_gpu_blocks]
            df_policy = PolicyFactory.get_policy(policy_name="sdf")
            remaining_running, running_scheduled_recorder = self._estimate_schedule_running(
                remaining_running,
                budget,
                curr_loras,
                df_policy,
                enable_chunking=True)
            gpu_consume.append(self.free_gpu_blocks)

            # Schedule swapped out requests.
            # If preemption happens, it means we don't have space for swap-in.
            is_running_preempted = any([item.lazy_state == LazyStates.RECOMPUTE or item.lazy_state == LazyStates.SWAP for item in running_scheduled_recorder])
            if not is_running_preempted:
                remaining_swapped, swapped_in_recorder = self._estimate_schedule_swapped(
                    remaining_swapped, budget, curr_loras, df_policy)
            else:
                swapped_in_recorder = deque()
            gpu_consume.append(self.free_gpu_blocks)

            # Schedule new prefills.
            remaining_waiting, prefills_recorder = self._estimate_schedule_prefills(
                remaining_waiting, budget, curr_loras, enable_chunking=True)
            gpu_consume.append(self.free_gpu_blocks)

            assert (budget.num_batched_tokens <=
                    self.scheduler_config.max_num_batched_tokens)
            assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

            # Update waiting requests.
            self.waiting = remaining_waiting
            # Update new running requests.
            self.running = remaining_running
            # Update swapped requests.
            self.swapped = remaining_swapped

            if len(running_scheduled_recorder) + len(prefills_recorder) + len(swapped_in_recorder) > 0:
                # update computed number in advance
                scheduled_recorders = (running_scheduled_recorder + 
                                        prefills_recorder + 
                                        swapped_in_recorder)
                for s in scheduled_recorders:
                    # do_sample
                    s.do_sample = not (s.is_prefill() and 
                        s.num_new_tokens + s.new_data_num_computed_tokens[0] < s.new_data_len[0])

                    # pre-execute update_num_computed_tokens
                    s.new_data_num_computed_tokens = [s.new_data_num_computed_tokens[0] + s.num_new_tokens]
                    num_computed_tokens = s.new_data_num_computed_tokens[0]
                    data_len = s.new_data_len[0]
                    assert num_computed_tokens <= data_len, (
                        num_computed_tokens, data_len)
                    if data_len == num_computed_tokens:
                        s.new_data_stages = [SequenceStage.DECODE]
                    
                    # pre-execute process_prompt_logprob
                    # if (s.new_data_stages[0] == SequenceStage.DECODE and 
                    #     s.new_data_len[0] == s.new_data_num_computed_tokens[0]):
                    if s.do_sample:
                        s.new_data_len = [s.new_data_len[0] + 1]
                    # save current state, will be checked after 2 iterations
                    s.history.append([s.new_data_num_computed_tokens, s.new_data_stages,
                                      s.new_data_len, s.num_new_tokens, s.new_seq_states,
                                      s.do_sample,])
                
                # wrap sequence groups into running
                # NOTE: we should not update in-place
                into_running = ([item for item in prefills_recorder if item.lazy_state == LazyStates.PREFILL] +
                                [item for item in running_scheduled_recorder if item.lazy_state == LazyStates.DECODE] +
                                [item for item in running_scheduled_recorder if item.lazy_state == LazyStates.PREFILL] +
                                [item for item in swapped_in_recorder if item.lazy_state == LazyStates.DECODE] +
                                [item for item in swapped_in_recorder if item.lazy_state == LazyStates.PREFILL])
                into_running = [
                    LazySequenceGroup(
                        item.seq_group,
                        new_data_num_computed_tokens=item.new_data_num_computed_tokens,
                        new_data_stages=item.new_data_stages,
                        new_data_len=item.new_data_len,
                        num_new_tokens=0,
                        new_seq_states=[SequenceStatus.RUNNING],
                        history=item.history,
                    ) for item in into_running
                ]

                # Update waiting requests.
                self.waiting.extendleft([
                    LazySequenceGroup(
                        item.seq_group,
                        new_data_num_computed_tokens=item.new_data_num_computed_tokens,
                        new_data_stages=item.new_data_stages,
                        new_data_len=item.new_data_len,
                        num_new_tokens=0,
                        new_seq_states=[SequenceStatus.WAITING],
                        history=item.history,
                    ) for item in running_scheduled_recorder
                        if item.lazy_state == LazyStates.RECOMPUTE
                ])#(running_scheduled.preempted)
                # self.waiting.extendleft(decode_preempted.preempted)

                # Update new running requests.
                self.running.extend(into_running)
                # self.running.extend([s.seq_group for s in prefills.seq_groups])
                # self.running.extend(
                #     [s.seq_group for s in running_scheduled.decode_seq_groups])
                # self.running.extend(
                #     [s.seq_group for s in running_scheduled.prefill_seq_groups])
                # self.running.extend(
                #     [s.seq_group for s in swapped_in.decode_seq_groups])
                # self.running.extend(
                #     [s.seq_group for s in swapped_in.prefill_seq_groups])

                # Update swapped requests.
                self.swapped.extend([
                    LazySequenceGroup(
                        item.seq_group,
                        new_seq_states=[SequenceStatus.SWAPPED],
                        history=item.history,
                    ) for item in running_scheduled_recorder
                        if item.lazy_state == LazyStates.SWAP
                ])#(running_scheduled.swapped_out)
                # self.swapped.extend(decode_preempted.swapped_out)

                # maybe new requests into waiting during scheduling
                assert self.get_num_running_requests() == total_len

                self.gpu_consume_records.append(gpu_consume)

                self.recorder_queue.put([
                    running_scheduled_recorder,
                    prefills_recorder,
                    swapped_in_recorder,
                    budget])
            else:
                logger.warning("Nothing in recorder, put one token")
                self.scheduling_count -= 1
                self.token_queue.put(token)
        # except Exception as e:
        #     logger.warning(f"Error in lazy scheduling: {e}")
        #     self.scheduling_count -= 1
        #     self.token_queue.put(token)
        finally:
            t1 = time.time()
            self.atomic.release()
        
        return t1 - t0

    def _preempt_executor(
        self, 
        seq_group: SequenceGroup, 
        old_states: List[SequenceStatus],
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: PreemptionMode
    ) -> None:
        if preemption_mode == PreemptionMode.RECOMPUTE:
            for seq, seq_old_state in zip(seq_group.get_seqs(), old_states):
                    if seq_old_state == SequenceStatus.RUNNING:
                        self.free_seq(seq)
        elif preemption_mode == PreemptionMode.SWAP:
            # mapping = self.block_manager.swap_in(seq_group)
            request_id = seq_group.request_id

            # CPU block -> GPU block.
            # dict is efficient in lookup `if cpu_block in mapping`
            mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            for seq, seq_old_state in zip(seq_group.get_seqs(), old_states):
                if seq_old_state == SequenceStatus.SWAPPED:
                    self.block_manager.block_tables[seq.seq_id] = \
                        self.block_manager._swap_block_table(self.block_manager.block_tables[seq.seq_id],
                                            self.block_manager.cpu_allocator,
                                            self.block_manager.gpu_allocator,
                                            mapping)

            if seq_group.is_encoder_decoder():
                self.cross_block_tables[request_id] = \
                    self._swap_block_table(self.cross_block_tables[request_id],
                                        self.cpu_allocator,
                                        self.gpu_allocator,
                                        mapping)
            blocks_to_swap_out.extend(mapping)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _lazy_running_executor(
        self,
        running_scheduled_recorder: deque,
    ) -> SchedulerRunningOutputs:
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroupV2] = []
        prefill_seq_groups: List[ScheduledSequenceGroupV2] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        for item in running_scheduled_recorder:
            state, num_new_tokens, do_sample = \
                item.lazy_state, item.num_new_tokens, item.do_sample
            seq_group = item.seq_group

            if state == LazyStates.RECOMPUTE:
                self._preempt(seq_group, blocks_to_swap_out, PreemptionMode.RECOMPUTE)
                preempted.append(seq_group)
            elif state == LazyStates.SWAP:
                self._preempt(seq_group, blocks_to_swap_out, PreemptionMode.SWAP)
                swapped_out.append(seq_group)
            elif state == LazyStates.PREFILL:
                assert num_new_tokens > 0
                self._append_slots(seq_group, blocks_to_copy)
                # assert num_new_tokens + seq_group.get_seqs()[0].data.get_num_computed_tokens() >= \
                #     seq_group.get_seqs()[0].data.get_len(), \
                #     f"{num_new_tokens} + {seq_group.get_seqs()[0].data.get_num_computed_tokens()} >=" + \
                #     f"{seq_group.get_seqs()[0].data.get_len()}"
                prefill_seq_groups.append(
                        ScheduledSequenceGroupV2(
                            seq_group=seq_group,
                            token_chunk_size=num_new_tokens,
                            do_sample=do_sample))
            elif state == LazyStates.DECODE:
                assert num_new_tokens == 1
                self._append_slots(seq_group, blocks_to_copy)
                # assert num_new_tokens + seq_group.get_seqs()[0].data.get_num_computed_tokens() >= \
                #     seq_group.get_seqs()[0].data.get_len(), \
                #     f"{num_new_tokens} + {seq_group.get_seqs()[0].data.get_num_computed_tokens()} >=" + \
                #     f"{seq_group.get_seqs()[0].data.get_len()}"
                decode_seq_groups.append(
                        ScheduledSequenceGroupV2(seq_group=seq_group,
                                               token_chunk_size=1,
                                               do_sample=do_sample))
            else:
                assert False, "state out of expected cases"

        return SchedulerRunningOutputs(
                decode_seq_groups=decode_seq_groups,
                prefill_seq_groups=prefill_seq_groups,
                preempted=preempted,
                swapped_out=swapped_out,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                num_lookahead_slots=0)
    
    def _lazy_swapped_executor(
        self,
        swapped_scheduled_recorder: LazySchedule,
    ) -> SchedulerSwappedInOutputs:
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroupV2] = []
        prefill_seq_groups: List[ScheduledSequenceGroupV2] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        for item in swapped_scheduled_recorder:
            state, num_new_tokens, do_sample = \
                item.lazy_state, item.num_new_tokens, item.do_sample
            seq_group = item.seq_group

            if state == LazyStates.NEVER:
                infeasible_seq_groups.append(seq_group)
            elif state == LazyStates.PREFILL:
                assert num_new_tokens > 0
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slots(seq_group, blocks_to_copy)
                # assert num_new_tokens + seq_group.get_seqs()[0].data.get_num_computed_tokens() >= \
                #     seq_group.get_seqs()[0].data.get_len(), \
                #     f"{num_new_tokens} + {seq_group.get_seqs()[0].data.get_num_computed_tokens()} >=" + \
                #     f"{seq_group.get_seqs()[0].data.get_len()}"
                prefill_seq_groups.append(
                    ScheduledSequenceGroupV2(seq_group,
                                           token_chunk_size=num_new_tokens,
                                           do_sample=do_sample))
            elif state == LazyStates.DECODE:
                assert num_new_tokens == 1
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slots(seq_group, blocks_to_copy)
                # assert num_new_tokens + seq_group.get_seqs()[0].data.get_num_computed_tokens() >= \
                #     seq_group.get_seqs()[0].data.get_len(), \
                #     f"{num_new_tokens} + {seq_group.get_seqs()[0].data.get_num_computed_tokens()} >=" + \
                #     f"{seq_group.get_seqs()[0].data.get_len()}"
                decode_seq_groups.append(
                    ScheduledSequenceGroupV2(seq_group, token_chunk_size=1,
                                             do_sample=do_sample))
            else:
                assert False, "state out of expected cases"

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=0,
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _lazy_prefill_executor(
        self,
        prefill_scheduled_recorder: LazySchedule,
    ) -> SchedulerPrefillOutputs:
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroupV2] = []

        for item in prefill_scheduled_recorder:
            state, num_new_tokens, do_sample = \
                item.lazy_state, item.num_new_tokens, item.do_sample
            seq_group = item.seq_group

            if state == LazyStates.IGNORE:
                ignored_seq_groups.append(seq_group)
            elif state == LazyStates.PREFILL:
                assert num_new_tokens > 0
                self._allocate_and_set_running(seq_group)
                # assert num_new_tokens + seq_group.get_seqs()[0].data.get_num_computed_tokens() >= \
                #     seq_group.get_seqs()[0].data.get_len(), \
                #     f"{num_new_tokens} + {seq_group.get_seqs()[0].data.get_num_computed_tokens()} >=" + \
                #     f"{seq_group.get_seqs()[0].data.get_len()}"
                seq_groups.append(
                    ScheduledSequenceGroupV2(seq_group=seq_group,
                                        token_chunk_size=num_new_tokens,
                                        do_sample=do_sample))
            else:
                assert False, "state out of expected cases"
                
        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=0)

    def _lazy_executor(self) -> SchedulerOutputs:
        # logger.info("lazy executor runs.")

        if self.is_sync:
            if self.token_queue.empty():
                self.token_queue.put_nowait(f"exceeding_{self.scheduling_count-self.execution_count}")
                # logger.info(f"{self.execution_count} sync running runs.")
                self.execution_count += 1
            logger.warning("Waiting for sync processed.")
            self.sync_event.wait()
            self.sync_event.clear()

        try:
            running_scheduled_recorder, prefills_recorder, swapped_in_recorder, budget = \
                self.recorder_queue.get(timeout=10)#(timeout=0.01)
        except Exception as e:
            logger.warning("Lazy executor time out.")
            return SchedulerOutputs(
                scheduled_seq_groups=[],
                num_prefill_groups=0,
                num_batched_tokens=0,
                blocks_to_swap_in=[],
                blocks_to_swap_out=[],
                blocks_to_copy=[],
                ignored_seq_groups=[],
                num_lookahead_slots=0,
                running_queue_size=0,
                preempted=0,
            )

        self.atomic.acquire()
        t0 = time.time()
        estimated_gpu_consume = self.gpu_consume_records.pop(0)
        actual_gpu_consume = [self.block_manager.gpu_allocator.get_num_free_blocks()]
        decode_preempted = SchedulerRunningOutputs.create_empty()
        running_scheduled = self._lazy_running_executor(running_scheduled_recorder)
        actual_gpu_consume.append(self.block_manager.gpu_allocator.get_num_free_blocks())
        swapped_in = self._lazy_swapped_executor(swapped_in_recorder)
        actual_gpu_consume.append(self.block_manager.gpu_allocator.get_num_free_blocks())
        prefills = self._lazy_prefill_executor(prefills_recorder)
        actual_gpu_consume.append(self.block_manager.gpu_allocator.get_num_free_blocks())
        self.cache_consumption = time.time() - t0
        self.atomic.release()

        # validate gpu consumption
        executor_name = ['init', 'running', 'swapped', 'prefill']
        for i, (egc, agc) in enumerate(zip(estimated_gpu_consume, actual_gpu_consume)):
            if egc != agc:
                logger.warning(f"gpu block estimation error in {executor_name[i]}: {egc} != {agc}")
                self.free_gpu_blocks += (actual_gpu_consume[-1] - estimated_gpu_consume[-1])
                break
        # self.free_gpu_blocks += (actual_gpu_consume[-1] - estimated_gpu_consume[-1])

        # update metrics
        self._update_time_metrcis(prefills, running_scheduled, 
                                swapped_in, decode_preempted)

        # sync, seq_group is updated except for computed tokens and data len (output)
        if self.token_queue.empty():
            self.token_queue.put_nowait(f"exceeding_{self.scheduling_count-self.execution_count}")
            # logger.info(f"{self.execution_count} sync running runs.")
            self.execution_count += 1
        
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + 
                            decode_preempted.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
                        swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) + 
                    len(decode_preempted.preempted)),
        )

    def sync_running(self) -> None:
        # run after one iteration
        for item in set(self.running + self.waiting + self.swapped):
            if type(item) == LazySequenceGroup:
                # check correct
                # seq_group = item.seq_group.get_seqs()[0]
                # assert item.new_seq_states[0] == seq_group.status, \
                #     f"{item.new_seq_states[0]} != {seq_group.status}"
                # assert item.new_data_stages[0] == seq_group.data.stage, \
                #     f"{item.new_data_stages[0]} != {seq_group.data.stage}"
                # assert item.new_data_num_computed_tokens[0] == \
                #     seq_group.data.get_num_computed_tokens(), \
                #     f"{item.new_data_num_computed_tokens[0]} != " + \
                #     f"{seq_group.data.get_num_computed_tokens()}"
                # assert item.new_data_len[0] == seq_group.data.get_len(), \
                #     f"{item.new_data_len[0]} != {seq_group.data.get_len()}"

                item.update_by_self()

        # assert self.scheduling_count - self.execution_count <= 1, \
        #     f"{self.execution_count}, {self.scheduling_count}"
        if self.token_queue.empty():
            self.token_queue.put(f"exceeding_{self.scheduling_count-self.execution_count}", timeout=1)
            # logger.info(f"{self.execution_count} sync running runs.")
            self.execution_count += 1

    def check_sync(self) -> None:
        # assert all([item.seq_group.get_seqs()[0].data.get_num_computed_tokens() == 
        #             item.seq_group.get_seqs()[0].data.get_len() 
        #             for item in self.running + self.waiting + self.swapped])
        for item in set(self.running + self.waiting + self.swapped):
            if type(item) == LazySequenceGroup:
                assert len(item.history) <= 3, f"{len(item.history)}"
                # [new_data_num_computed_tokens, new_data_stages,
                #   new_data_len, num_new_tokens, new_seq_states,]
                history = item.history[0]
                # check correct
                seq_group = item.seq_group.get_seqs()[0]
                is_error = any([
                    history[0][0] != seq_group.data.get_num_computed_tokens(),
                    history[1][0] != seq_group.data.stage,
                    history[2][0] != seq_group.data.get_len(),
                    history[4][0] != seq_group.status,
                ])
                if is_error:
                    logger.warning(f"Sequence group estimation error, raise sync.")
                    self.cache_lock()
                    self.is_sync = True
                    self.cache_unlock()
                    break
                item.history.popleft()
                # assert history[0][0] == \
                #     seq_group.data.get_num_computed_tokens(), \
                #     f"{history[0][0]} != " + \
                #     f"{seq_group.data.get_num_computed_tokens()}"
                # assert history[1][0] == seq_group.data.stage, \
                #     f"{history[1][0]} != {seq_group.data.stage}"
                # assert history[2][0] == seq_group.data.get_len(), \
                #     f"{history[2][0]} != {seq_group.data.get_len()}"
                # assert history[4][0] == seq_group.status, \
                #     f"{history[4][0]} != {seq_group.status}"
                    
    def schedule_restore(self) -> None:
        while not self.recorder_queue.empty():
            self.recorder_queue.get_nowait()

        waiting_groups = [item.seq_group if type(item) == LazySequenceGroup else item for item in self.waiting ]
        running_groups = [item.seq_group if type(item) == LazySequenceGroup else item for item in self.running ]
        swapped_groups = [item.seq_group if type(item) == LazySequenceGroup else item for item in self.swapped ]

        waiting_groups = [item for item in waiting_groups+running_groups+swapped_groups if item.get_seqs()[0].status == SequenceStatus.WAITING]
        running_groups = [item for item in waiting_groups+running_groups+swapped_groups if item.get_seqs()[0].status == SequenceStatus.RUNNING]
        swapped_groups = [item for item in waiting_groups+running_groups+swapped_groups if item.get_seqs()[0].status == SequenceStatus.SWAPPED]

        self.waiting = waiting_groups
        self.running = running_groups
        self.swapped = swapped_groups

        self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()

    def check_estimation(
        self,
        seq_group: SequenceGroup,
        lazy_seq: LazySequenceGroup,
    )-> bool:
        seq = seq_group.get_seqs()[0].data
        return (lazy_seq.new_data_num_computed_tokens[0] == seq.get_num_computed_tokens() and
            lazy_seq.new_data_stages[0] == seq.stage and
            lazy_seq.new_data_len[0] == seq.get_len() and
            lazy_seq.new_seq_states[0] == seq.status)

    def free_finished_seq_groups(self) -> int:
        # finished = [seq_group for seq_group in self.running if seq_group.is_finished()]
        # if len(finished) > 0:
        #     logger.info(f"Free {len(finished)} finished seq groups")
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())
        return len(self.running)
    
    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.atomic.acquire()
        self.waiting.append(seq_group)
        self.atomic.release()

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        self.atomic.acquire()
        super().abort_seq_group(request_id)
        self.atomic.release()

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()
        now = time.time()

        # log schedule gap
        # logger.info("###Debug: schedule gap %f###", now-self.last_schedule_time)
        self.last_schedule_time = now

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

                # context_len = seq.data.get_len() - 1
                # seq_len = min(
                #     seq.data.get_len(),
                #     context_len + token_chunk_size)
                # block_size = self.block_manager.block_size
                # assert len(block_tables[seq_id]) - 1 >= (seq_len - 1) // block_size

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            # do_sample = scheduled_seq_group.do_sample
            # assert len(seq_data) > 0
            do_sample = True
            if seq_group.is_prefill():
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + seqs[0].data.get_num_computed_tokens() <
                        seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            is_prompt = seq_group.is_prefill()
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                do_sample=do_sample,
                pooling_params=seq_group.pooling_params,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.num_prefill_groups > 0 else None,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)

        return seq_group_metadata_list, scheduler_outputs

    def cache_lock(self) -> None:
        self.atomic.acquire()

    def cache_unlock(self) -> None:
        self.atomic.release()

class LazySchedulerV1(SchedulerV2):
    def __init__(self, 
                scheduler_config: SchedulerConfig,
                cache_config: CacheConfig,
                lora_config: Optional[LoRAConfig]) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
        # pipeline
        self.recorder_queue: Queue[LazySchedule, LazySchedule, LazySchedule, SchedulingBudget] = Queue(maxsize=1)

    def _schedule(self) -> SchedulerOutputs:
        # return self._schedule_chunked_prefill_with_predicted_length()
        # return self._test_schedule_chunked_prefill_with_predicted_length()
        return self._lazy_executor()

    def _test_schedule_chunked_prefill_with_predicted_length(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        total_seq_groups = len(self.waiting) + len(self.running) + len(self.swapped)

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                    SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        # self.running, decode_preempted = self._update_running_decode(
        #     self.running,
        #     budget,
        #     curr_loras,
        #     enable_chunking=True)
        self._update_remaining_decode(self.running)
        decode_preempted = SchedulerRunningOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        # fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        df_policy = PolicyFactory.get_policy(policy_name="sdf")

        # test lazy
        self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()

        # fork_budget = copy.deepcopy(budget)
        # fork_running = copy.deepcopy(self.running)
        # estimated_remaing_running, estimated_running_scheduled = self._estimate_schedule_running(
        #     fork_running,
        #     fork_budget,
        #     curr_loras,
        #     df_policy,
        #     enable_chunking=True)

        remaining_running, running_scheduled = self._schedule_running(
            self.running,
            budget,
            curr_loras,
            df_policy,
            enable_chunking=True)

        # assert set([str(s.seq_group) for s in running_scheduled.decode_seq_groups]) == \
        #     set([str(s.seq_group) for s in estimated_running_scheduled if s.state == LazyStates.DECODE])
        # assert set([str(s.seq_group) for s in running_scheduled.prefill_seq_groups]) == \
        #     set([str(s.seq_group) for s in estimated_running_scheduled if s.state == LazyStates.PREFILL])
        # assert set([str(s) for s in running_scheduled.preempted]) == \
        #     set([str(s.seq_group) for s in estimated_running_scheduled if s.state == LazyStates.RECOMPUTE])
        # assert set([str(s) for s in running_scheduled.swapped_out]) == \
        #     set([str(s.seq_group) for s in estimated_running_scheduled if s.state == LazyStates.SWAP])
        # assert set([str(s) for s in estimated_remaing_running]) == set([str(s) for s in remaining_running])
        # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) + len(
                    decode_preempted.swapped_out) + len(
                        decode_preempted.preempted) == 0:
            # self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
            # fork_budget = copy.deepcopy(budget)
            # fork_swapped = copy.deepcopy(self.swapped)
            # estimated_remaining_swapped, estimated_swapped_in = self._estimate_schedule_swapped(
            #     fork_swapped,
            #     fork_budget,
            #     curr_loras,
            #     df_policy)

            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, df_policy)
            
            # assert set([str(s) for s in remaining_swapped]) == set([str(s) for s in estimated_remaining_swapped])
            # assert set([str(s.seq_group) for s in swapped_in.decode_seq_groups]) == \
            #     set([str(s.seq_group) for s in estimated_swapped_in if s.state == LazyStates.DECODE])
            # assert set([str(s.seq_group) for s in swapped_in.prefill_seq_groups]) == \
            #     set([str(s.seq_group) for s in estimated_swapped_in if s.state == LazyStates.PREFILL])
            # assert set([str(s) for s in swapped_in.infeasible_seq_groups]) == \
            #     set([str(s.seq_group) for s in estimated_swapped_in if s.state == LazyStates.NEVER])
            # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()
            

        # Schedule new prefills.
        # self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
        # fork_budget = copy.deepcopy(budget)
        # fork_waiting = copy.deepcopy(self.waiting)
        # estimated_remaining_waiting, estimated_prefills = self._estimate_schedule_prefills(
        #     fork_waiting,
        #     fork_budget,
        #     curr_loras,
        #     enable_chunking=True)

        remaining_waiting, prefills = self._schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)
        
        # assert set([str(s) for s in estimated_remaining_waiting]) == set([str(s) for s in remaining_waiting])
        # assert set([str(s.seq_group) for s in prefills.seq_groups]) == \
        #     set([str(s.seq_group) for s in estimated_prefills if s.state == LazyStates.PREFILL])
        # assert set([str(s) for s in prefills.ignored_seq_groups]) == \
        #     set([str(s.seq_group) for s in estimated_prefills if s.state == LazyStates.IGNORE])
        # assert self.free_gpu_blocks == self.block_manager.gpu_allocator.get_num_free_blocks()

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        self.waiting.extendleft(decode_preempted.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        self.swapped.extend(decode_preempted.swapped_out)
        # update metrics
        assert total_seq_groups == len(self.waiting) + len(self.running) + len(self.swapped)
        self._update_time_metrcis(prefills, running_scheduled, 
                                swapped_in, decode_preempted)

        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + 
                            decode_preempted.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
                        swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) + 
                    len(decode_preempted.preempted)),
        )

    def _maybe_can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: Optional[int] = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation not supported in BlockSpaceManagerV1"

        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= self.free_gpu_blocks

    def _lazy_preempt(
        self,
        seq_group: SequenceGroup,
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        if preemption_mode is None:
            if seq_group.remaining_decode >= 1:
                # prioritize remaining decode
                preemption_mode = PreemptionMode.SWAP
            elif seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        # physical_block = self.block_manager._get_physical_blocks(seq_group)
        # self.free_gpu_blocks += len(physical_block)

        if preemption_mode == PreemptionMode.RECOMPUTE:
            physical_block = self.block_manager._get_physical_blocks(seq_group)
            self.free_gpu_blocks += len(physical_block)
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            assert len(seqs) == 1
            for seq in seqs:
                seq.status = SequenceStatus.WAITING
                seq.reset_state_for_recompute()
        else:
            if not self.block_manager.can_swap_out(seq_group):
                # FIXME(woosuk): Abort the sequence group instead of aborting the
                # entire engine.
                raise RuntimeError(
                    "Aborted due to the lack of CPU swap space. Please increase "
                    "the swap space to avoid this error.")
            physical_block = self.block_manager._get_physical_blocks(seq_group)
            self.free_gpu_blocks += len(physical_block)
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq.status = SequenceStatus.SWAPPED

        return preemption_mode

    def _lazy_append_slots(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            logical_blocks = seq.logical_token_blocks
            block_table = self.block_manager.block_tables[seq.seq_id]
            if len(block_table) < len(logical_blocks):
                # Currently this code only supports adding one physical block
                assert len(block_table) == len(logical_blocks) - 1
                self.free_gpu_blocks -= 1

    def _estimate_schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, deque]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        assert not self.lora_enabled, 'curr_loras is not supported in lazy scheduling'
        # Record operations that are executed later.
        recorder: Deque[LazySchedule] = deque()

        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)
        # running_queue = self.running

        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            while not self._maybe_can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                t0 = time.time()
                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()
                    preempted_mode = self._lazy_preempt(victim_seq_group)
                    
                    # preempted_mode = self._preempt(victim_seq_group,
                    #                                blocks_to_swap_out)
                    old_states = [s.status for s in victim_seq_group.get_seqs()]
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        # preempted.append(victim_seq_group)
                        recorder.append(LazySchedule(victim_seq_group, LazyStates.RECOMPUTE, 
                                                     old_states=old_states, num_running_tokens=0))
                    else:
                        # swapped_out.append(victim_seq_group)
                        recorder.append(LazySchedule(victim_seq_group, LazyStates.SWAP,
                                                     old_states=old_states, num_running_tokens=0))
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._lazy_preempt(seq_group)
                    # preempted_mode = self._preempt(seq_group,
                    #                                blocks_to_swap_out)
                    old_states = [s.status for s in seq_group.get_seqs()]
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        # preempted.append(seq_group)
                        recorder.append(LazySchedule(seq_group, LazyStates.RECOMPUTE,
                                                     old_states=old_states, num_running_tokens=0))
                    else:
                        # swapped_out.append(seq_group)
                        recorder.append(LazySchedule(seq_group, LazyStates.SWAP,
                                                     old_states=old_states, num_running_tokens=0))
                    break
            else:
                t0 = time.time()
                # self._append_slots(seq_group, blocks_to_copy)
                old_states = [s.status for s in seq_group.get_seqs()]
                self._lazy_append_slots(seq_group)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    # prefill_seq_groups.append(
                    #     ScheduledSequenceGroup(
                    #         seq_group=seq_group,
                    #         token_chunk_size=num_running_tokens))
                    recorder.append(LazySchedule(seq_group, LazyStates.PREFILL, 
                                                 old_states, num_running_tokens))
                else:
                    # decode_seq_groups.append(
                    #     ScheduledSequenceGroup(seq_group=seq_group,
                    #                            token_chunk_size=1))
                    recorder.append(LazySchedule(seq_group, LazyStates.DECODE, 
                                                 old_states, 1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)

        return running_queue, recorder

    def _maybe_can_swap_in(self,
                    seq_group: SequenceGroup,
                    num_lookahead_slots: int = 0) -> AllocStatus:
        assert (num_lookahead_slots == 0
                ), "BlockSpaceManagerV1 does not support lookahead allocation"

        blocks = self.block_manager._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        if seq_group.is_encoder_decoder():
            num_swapped_seqs += 1
        num_free_blocks = self.free_gpu_blocks
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        if self.block_manager.gpu_allocator.get_num_total_blocks() < num_required_blocks:
            return AllocStatus.NEVER
        elif num_free_blocks - num_required_blocks >= self.block_manager.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _lazy_swap_in(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            num_logical_blocks = len(seq.logical_token_blocks)
            self.free_gpu_blocks -= num_logical_blocks
            seq.status = SequenceStatus.RUNNING

    def _estimate_schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, deque]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        assert not self.lora_enabled, 'curr_loras is not supported in lazy scheduling'
        # record operations that are executed later.
        recorder: Deque[LazySchedule] = deque()

        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)
        # swapped_queue = self.swapped

        while swapped_queue:
            seq_group = swapped_queue[0]
            old_states = [s.status for s in seq_group.get_seqs()]

            # If the sequence group cannot be swapped in, stop.
            alloc_status = self._maybe_can_swap_in(seq_group)
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                # infeasible_seq_groups.append(seq_group)
                recorder.append(LazySchedule(seq_group, LazyStates.NEVER,
                                             old_states, 0))
                swapped_queue.popleft()
                continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            swapped_queue.popleft()

            t0 = time.time()
            # self._swap_in(seq_group, blocks_to_swap_in)
            # self._append_slots(seq_group, blocks_to_copy)
            self._lazy_swap_in(seq_group)
            self._lazy_append_slots(seq_group)
            self.cache_consumption += time.time() - t0

            is_prefill = seq_group.is_prefill()
            if is_prefill:
                # prefill_seq_groups.append(
                #     ScheduledSequenceGroup(seq_group,
                #                            token_chunk_size=num_new_tokens))
                recorder.append(LazySchedule(seq_group, LazyStates.PREFILL, 
                                             old_states, num_new_tokens))
            else:
                # decode_seq_groups.append(
                #     ScheduledSequenceGroup(seq_group, token_chunk_size=1))
                recorder.append(LazySchedule(seq_group, LazyStates.DECODE, 
                                             old_states, 1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        return swapped_queue, recorder

    def _maybe_can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self.block_manager, seq_group)

        self_num_required_blocks = self.block_manager._get_seq_num_required_blocks(
            seq_group.get_seqs(status=SequenceStatus.WAITING)[0])
        cross_num_required_blocks = self.block_manager._get_seq_num_required_blocks(
            seq_group.get_encoder_seq())
        num_required_blocks = self_num_required_blocks + \
                              cross_num_required_blocks

        if self.block_manager.block_sliding_window is not None:

            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.free_gpu_blocks

        # Use watermark to avoid frequent cache eviction.
        if (self.block_manager.num_total_gpu_blocks - num_required_blocks <
                self.block_manager.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.block_manager.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _lazy_allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        is_encoder_decoder = seq_group.is_encoder_decoder()
        check_no_caching_or_swa_for_blockmgr_encdec(self.block_manager, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_prompt_blocks = len(seq.logical_token_blocks)
        self.free_gpu_blocks -= num_prompt_blocks

        if is_encoder_decoder:
            seq_ = seq_group.get_encoder_seq()
            num_prompt_blocks_ = len(seq_.logical_token_blocks)
            self.free_gpu_blocks -= num_prompt_blocks_

        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _estimate_schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, deque]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        assert not self.lora_enabled, 'curr_loras is not supported in lazy scheduling'
        waiting_queue = deque([s for s in waiting_queue])
        # record operations that are executed later.
        recorder: Deque[LazySchedule] = deque()

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            old_states = [s.status for s in seq_group.get_seqs()]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                # ignored_seq_groups.append(seq_group)
                recorder.append(LazySchedule(seq_group, LazyStates.IGNORE,
                                             old_states, 0))
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            # can_allocate = self.block_manager.can_allocate(seq_group)
            can_allocate = self._maybe_can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                # ignored_seq_groups.append(seq_group)
                recorder.append(LazySchedule(seq_group, LazyStates.IGNORE,
                                             old_states, 0))
                waiting_queue.popleft()
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            waiting_queue.popleft()

            t0 = time.time()
            # self._allocate_and_set_running(seq_group)
            self._lazy_allocate_and_set_running(seq_group)
            self.cache_consumption += time.time() - t0

            recorder.append(LazySchedule(seq_group, LazyStates.PREFILL, 
                                         old_states, num_new_tokens))
            # seq_groups.append(
            #     ScheduledSequenceGroup(seq_group=seq_group,
            #                            token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len([r for r in recorder if r.state == LazyStates.PREFILL]) > 0:
            self.prev_prompt = True

        return waiting_queue, recorder
    
    def lazy_schedule(self) -> None:
        self.free_gpu_blocks = self.block_manager.gpu_allocator.get_num_free_blocks()
        self.cache_consumption = 0
        self._lazy_schedule_chunked_prefill_with_predicted_length()
    
    def _schedule(self) -> SchedulerOutputs:
        return self._lazy_executor()

    def _lazy_schedule_chunked_prefill_with_predicted_length(
        self
    ) -> Tuple[LazySchedule, LazySchedule, LazySchedule, SchedulingBudget]:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting = self.waiting
        remaining_running = self.running
        remaining_swapped = self.swapped

        self._update_remaining_decode(self.running)
        decode_preempted = SchedulerRunningOutputs.create_empty()

        df_policy = PolicyFactory.get_policy(policy_name="sdf")
        remaining_running, running_scheduled_recorder = self._estimate_schedule_running(
            self.running,
            budget,
            curr_loras,
            df_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        is_running_preempted = any([item.state == LazyStates.RECOMPUTE or item.state == LazyStates.SWAP for item in running_scheduled_recorder])
        if not is_running_preempted:
            remaining_swapped, swapped_in_recorder = self._estimate_schedule_swapped(
                self.swapped, budget, curr_loras, df_policy)
        else:
            swapped_in_recorder = deque()

        # Schedule new prefills.
        remaining_waiting, prefills_recorder = self._estimate_schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        # Update new running requests.
        self.running = remaining_running
        # Update swapped requests.
        self.swapped = remaining_swapped

        if len(running_scheduled_recorder) + len(prefills_recorder) + len(swapped_in_recorder) > 0:
            # update computed number in advance
            scheduled_recorders = (running_scheduled_recorder + 
                                   prefills_recorder + 
                                   swapped_in_recorder)
            for r in scheduled_recorders:
                if r.state == LazyStates.DECODE or r.state == LazyStates.PREFILL:
                    r.seq_group.update_num_computed_tokens(
                        r.num_running_tokens)
                
            self.recorder_queue.put([
                running_scheduled_recorder,
                prefills_recorder,
                swapped_in_recorder,
                budget])

    def _preempt_executor(
        self, 
        seq_group: SequenceGroup, 
        old_states: List[SequenceStatus],
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: PreemptionMode
    ) -> None:
        if preemption_mode == PreemptionMode.RECOMPUTE:
            for seq, seq_old_state in zip(seq_group.get_seqs(), old_states):
                    if seq_old_state == SequenceStatus.RUNNING:
                        self.free_seq(seq)
        elif preemption_mode == PreemptionMode.SWAP:
            # mapping = self.block_manager.swap_in(seq_group)
            request_id = seq_group.request_id

            # CPU block -> GPU block.
            # dict is efficient in lookup `if cpu_block in mapping`
            mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
            for seq, seq_old_state in zip(seq_group.get_seqs(), old_states):
                if seq_old_state == SequenceStatus.SWAPPED:
                    self.block_manager.block_tables[seq.seq_id] = \
                        self.block_manager._swap_block_table(self.block_manager.block_tables[seq.seq_id],
                                            self.block_manager.cpu_allocator,
                                            self.block_manager.gpu_allocator,
                                            mapping)

            if seq_group.is_encoder_decoder():
                self.cross_block_tables[request_id] = \
                    self._swap_block_table(self.cross_block_tables[request_id],
                                        self.cpu_allocator,
                                        self.gpu_allocator,
                                        mapping)
            blocks_to_swap_out.extend(mapping)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _lazy_running_executor(
        self,
        running_scheduled_recorder: LazySchedule,
    ) -> SchedulerRunningOutputs:
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        for item in running_scheduled_recorder:
            seq_group, state, old_states, num_running_tokens = item.seq_group, item.state, item.old_states, item.num_running_tokens
            cur_states = [s.status for s in seq_group.get_seqs()]
            for seq, _seq_state in zip(seq_group.get_seqs(), old_states):
                seq.status = _seq_state
            if state == LazyStates.RECOMPUTE:
                self._preempt(seq_group, blocks_to_swap_out, PreemptionMode.RECOMPUTE)
                preempted.append(seq_group)
            elif state == LazyStates.SWAP:
                self._preempt(seq_group, blocks_to_swap_out, PreemptionMode.SWAP)
                swapped_out.append(seq_group)
            elif state == LazyStates.PREFILL:
                self._append_slots(seq_group, blocks_to_copy)
                prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
            elif state == LazyStates.DECODE:
                self._append_slots(seq_group, blocks_to_copy)
                decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
            else:
                assert False, "state out of expected cases"
            for seq, _seq_state in zip(seq_group.get_seqs(), cur_states):
                    seq.status = _seq_state
        
        return SchedulerRunningOutputs(
                decode_seq_groups=decode_seq_groups,
                prefill_seq_groups=prefill_seq_groups,
                preempted=preempted,
                swapped_out=swapped_out,
                blocks_to_swap_out=blocks_to_swap_out,
                blocks_to_copy=blocks_to_copy,
                num_lookahead_slots=0)
    
    def _lazy_swapped_executor(
        self,
        swapped_scheduled_recorder: LazySchedule,
    ) -> SchedulerSwappedInOutputs:
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        for item in swapped_scheduled_recorder:
            seq_group, state, old_states, num_new_tokens = item.seq_group, item.state, item.old_states, item.num_running_tokens
            cur_states = [s.status for s in seq_group.get_seqs()]
            for seq, _seq_state in zip(seq_group.get_seqs(), old_states):
                seq.status = _seq_state

            if state == LazyStates.NEVER:
                infeasible_seq_groups.append(seq_group)
            elif state == LazyStates.PREFILL:
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slots(seq_group, blocks_to_copy)
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            elif state == LazyStates.DECODE:
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slots(seq_group, blocks_to_copy)
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            else:
                assert False, "state out of expected cases"

            for seq, _seq_state in zip(seq_group.get_seqs(), cur_states):
                    seq.status = _seq_state

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=0,
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _lazy_prefill_executor(
        self,
        prefill_scheduled_recorder: LazySchedule,
    ) -> SchedulerPrefillOutputs:
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []

        for item in prefill_scheduled_recorder:
            seq_group, state, old_states, num_new_tokens = item.seq_group, item.state, item.old_states, item.num_running_tokens
            cur_states = [s.status for s in seq_group.get_seqs()]
            for seq, _seq_state in zip(seq_group.get_seqs(), old_states):
                seq.status = _seq_state

            if state == LazyStates.IGNORE:
                ignored_seq_groups.append(seq_group)
            elif state == LazyStates.PREFILL:
                self._allocate_and_set_running(seq_group)
                seq_groups.append(
                    ScheduledSequenceGroup(seq_group=seq_group,
                                        token_chunk_size=num_new_tokens))
            else:
                assert False, "state out of expected cases"

            for seq, _seq_state in zip(seq_group.get_seqs(), cur_states):
                    seq.status = _seq_state
                
        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=0)

    def _lazy_executor(self) -> SchedulerOutputs:
        t0 = time.time()
        running_scheduled_recorder, prefills_recorder, swapped_in_recorder, budget = \
            self.recorder_queue.get()
        decode_preempted = SchedulerRunningOutputs.create_empty()
        running_scheduled = self._lazy_running_executor(running_scheduled_recorder)
        swapped_in = self._lazy_swapped_executor(swapped_in_recorder)
        prefills = self._lazy_prefill_executor(prefills_recorder)
        self.cache_consumption += time.time() - t0

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        self.waiting.extendleft(decode_preempted.preempted)
        # Update new running requests.
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        self.swapped.extend(decode_preempted.swapped_out)
        # update metrics
        self._update_time_metrcis(prefills, running_scheduled, 
                                swapped_in, decode_preempted)
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + 
                            decode_preempted.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
                        swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) + 
                    len(decode_preempted.preempted)),
        )

class SchedulerV1(Scheduler):
    '''
    RECOMPUTE version'''
    def __init__(self, 
                scheduler_config: SchedulerConfig,
                cache_config: CacheConfig,
                lora_config: Optional[LoRAConfig]) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        # track schedule gap
        self.last_schedule_time: float = 0.0
        # length predictor
        self.length_predictor = RandomLength()

    def _schedule(self) -> SchedulerOutputs:
        return self._schedule_chunked_prefill_with_predicted_length()
    
    def _schedule_chunked_prefill_with_predicted_length(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        total_seq_groups = len(self.waiting) + len(self.running) + len(self.swapped)

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                    SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        # self.running, decode_preempted = self._update_running_decode(
        #     self.running,
        #     budget,
        #     curr_loras,
        #     enable_chunking=True)
        self._update_remaining_decode(self.running)
        decode_preempted = SchedulerRunningOutputs.create_empty()

        # Decoding should be always scheduled first by fcfs.
        # fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        df_policy = PolicyFactory.get_policy(policy_name="sdf")
        remaining_running, running_scheduled = self._schedule_running(
            self.running,
            budget,
            curr_loras,
            df_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) + len(
                    decode_preempted.swapped_out) + len(
                        decode_preempted.preempted) == 0:
            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, df_policy)

        # Schedule new prefills.
        remaining_waiting, prefills = self._schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        self.waiting.extendleft(decode_preempted.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        self.swapped.extend(decode_preempted.swapped_out)
        # update metrics
        assert total_seq_groups == len(self.waiting) + len(self.running) + len(self.swapped)
        self._update_time_metrcis(prefills, running_scheduled, 
                                swapped_in, decode_preempted)

        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + 
                            decode_preempted.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
                        swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) + 
                    len(decode_preempted.preempted)),
        )

    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.

        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)
        # running_queue = self.running

        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.add(seq_group.lora_int_id)

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))
        
    def _schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerSwappedInOutputs]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)
        # swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group, self._get_num_lookahead_slots(is_prefill))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return swapped_queue, SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _predict_length(self, scheduled_seq_groups: List[ScheduledSequenceGroup]) -> List[ScheduledSequenceGroup]:
        list_seq_group = [sch_seq_group.seq_group for sch_seq_group in scheduled_seq_groups]
        list_seq_group = self.length_predictor.predict(list_seq_group)
        for updated_seq_group, sch_seq_group in zip(list_seq_group, scheduled_seq_groups):
            sch_seq_group.seq_group = updated_seq_group
        return scheduled_seq_groups

    def _update_remaining_decode(
        self,
        running_queue: deque,
    ) -> None:
        for seq_group in running_queue:
            #     # =1: finish current decoding, swap out
            #     # >1: current decoding
            #     # =0, is_prefill=1: prefilling, 
            #     # >0, is_prefill=1: RECOMPUTE,
            #     # =0 and is_prefill=0: decoding first token
            if seq_group.remaining_decode == 1:
                seq_group.just_end = 1
                seq_group.remaining_decode = 0
            else:
                if seq_group.remaining_decode > 1:
                    seq_group.remaining_decode -= 1
                elif seq_group.is_prefill():
                    pass
                elif not seq_group.is_prefill() and seq_group.remaining_decode == 0:
                    self.length_predictor.predict_one(seq_group)
                    seq_group.just_end = 0
                    assert seq_group.remaining_decode >= 1
                else:
                    assert False, "remaining_decode out of expected cases"
        
class SchedulerV0(Scheduler):
    def __init__(self, 
                scheduler_config: SchedulerConfig,
                cache_config: CacheConfig,
                lora_config: Optional[LoRAConfig]) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        # track schedule gap
        self.last_schedule_time: float = 0.0
        # length predictor
        self.length_predictor = RandomLength()

    def _schedule(self) -> SchedulerOutputs:
        return self._schedule_chunked_prefill_with_predicted_length()
    
    def _schedule_chunked_prefill_with_predicted_length(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """
        total_seq_groups = len(self.waiting) + len(self.running) + len(self.swapped)

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                    SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        self.running, decode_preempted = self._update_running_decode(
            self.running,
            budget,
            curr_loras,
            enable_chunking=True)

        # Decoding should be always scheduled first by fcfs.
        # fcfs_policy = PolicyFactory.get_policy(policy_name="fcfs")
        df_policy = PolicyFactory.get_policy(policy_name="sdf")
        remaining_running, running_scheduled = self._schedule_running(
            self.running,
            budget,
            curr_loras,
            df_policy,
            enable_chunking=True)

        # Schedule swapped out requests.
        # If preemption happens, it means we don't have space for swap-in.
        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) + len(
                    decode_preempted.swapped_out) + len(
                        decode_preempted.preempted) == 0:
            remaining_swapped, swapped_in = self._schedule_swapped(
                self.swapped, budget, curr_loras, df_policy)

        # Schedule new prefills.
        remaining_waiting, prefills = self._schedule_prefills(
            self.waiting, budget, curr_loras, enable_chunking=True)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        self.waiting.extendleft(decode_preempted.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        self.swapped.extend(decode_preempted.swapped_out)
        # update metrics
        assert total_seq_groups == len(self.waiting) + len(self.running) + len(self.swapped)
        self._update_time_metrcis(prefills, running_scheduled, 
                                swapped_in, decode_preempted)

        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups +
                                running_scheduled.decode_seq_groups +
                                swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out + 
                            decode_preempted.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
                        swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                    len(running_scheduled.swapped_out) + 
                    len(decode_preempted.preempted)),
        )
        
    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.

        now = time.time()
        running_queue = policy.sort_by_priority(now, running_queue)
        # running_queue = self.running

        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.add(seq_group.lora_int_id)

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))
        
    def _schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerSwappedInOutputs]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        now = time.time()
        swapped_queue = policy.sort_by_priority(now, swapped_queue)
        # swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group, self._get_num_lookahead_slots(is_prefill))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return swapped_queue, SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[SequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _preempt_v0(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.remaining_decode >= 1:
                # prioritize remaining decode
                preemption_mode = PreemptionMode.SWAP
            elif seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode
    
    def _predict_length(self, scheduled_seq_groups: List[ScheduledSequenceGroup]) -> List[ScheduledSequenceGroup]:
        list_seq_group = [sch_seq_group.seq_group for sch_seq_group in scheduled_seq_groups]
        list_seq_group = self.length_predictor.predict(list_seq_group)
        for updated_seq_group, sch_seq_group in zip(list_seq_group, scheduled_seq_groups):
            sch_seq_group.seq_group = updated_seq_group
        return scheduled_seq_groups

    def _update_running_decode_v0(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        '''
        preempt all running seq_group
        '''
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []
        remaining_running: Deque[Sequence] = deque()

        while running_queue:
            seq_group = running_queue[0]
            #     # =1: finish current decoding, swap out
            #     # >1: current decoding
            #     # =0 and is_prefill=1: prefilling
            #     # =0 and is_prefill=0: decoding first token
            if seq_group.remaining_decode == 1:
                seq_group.remaining_decode = 0
            else:
                if seq_group.remaining_decode > 1:
                    seq_group.remaining_decode -= 1
                elif seq_group.is_prefill():
                    pass
                elif not seq_group.is_prefill() and seq_group.remaining_decode == 0:
                    self.length_predictor.predict_one(seq_group)
                    assert seq_group.remaining_decode >= 1
                else:
                    assert False, "remaining_decode out of expected cases"
                
                remaining_running.append(seq_group)
                running_queue.popleft()
                continue

            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()

            budget.subtract_num_batched_tokens(seq_group.request_id,
                                                num_running_tokens)
            num_running_seqs = seq_group.get_max_num_running_seqs()
            budget.subtract_num_seqs(seq_group.request_id,
                                        num_running_seqs)
            if curr_loras is not None and seq_group.lora_int_id > 0:
                curr_loras.remove(seq_group.lora_int_id)
                
            # No other sequence groups can be preempted.
            # Preempt the current sequence group.
            # NOTE: use SWAP because RECOMPUTE loses remaining decode
            # but remaining_decode=0, so finally use RECOMPUTE
            preempted_mode = self._preempt(seq_group,
                                            blocks_to_swap_out)
            if preempted_mode == PreemptionMode.RECOMPUTE:
                preempted.append(seq_group)
            else:
                swapped_out.append(seq_group)

        return remaining_running, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

    def _update_running_decode(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs]:
        '''
        preempt all running seq_group
        '''
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []
        remaining_running: Deque[Sequence] = deque()

        while running_queue:
            seq_group = running_queue[0]
            #     # =1: finish current decoding, swap out
            #     # >1: current decoding
            #     # =0 and is_prefill=1: prefilling
            #     # =0 and is_prefill=0: decoding first token
            if seq_group.remaining_decode == 1:
                seq_group.remaining_decode = 0
            else:
                if seq_group.remaining_decode > 1:
                    seq_group.remaining_decode -= 1
                elif seq_group.is_prefill() and seq_group.remaining_decode == 0:
                    assert seq_group.remaining_decode == 0
                elif not seq_group.is_prefill() and seq_group.remaining_decode == 0:
                    self.length_predictor.predict_one(seq_group)
                    assert seq_group.remaining_decode >= 1
                else:
                    assert False, "remaining_decode out of expected cases"
                
                remaining_running.append(seq_group)
                running_queue.popleft()
                continue

            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()

            budget.subtract_num_batched_tokens(seq_group.request_id,
                                                num_running_tokens)
            num_running_seqs = seq_group.get_max_num_running_seqs()
            budget.subtract_num_seqs(seq_group.request_id,
                                        num_running_seqs)
            if curr_loras is not None and seq_group.lora_int_id > 0:
                curr_loras.remove(seq_group.lora_int_id)
                
            # No other sequence groups can be preempted.
            # Preempt the current sequence group.
            # NOTE: use SWAP because RECOMPUTE loses remaining decode
            # but remaining_decode=0, so finally use RECOMPUTE
            preempted_mode = self._preempt(seq_group,
                                            blocks_to_swap_out)
            if preempted_mode == PreemptionMode.RECOMPUTE:
                preempted.append(seq_group)
            else:
                swapped_out.append(seq_group)

        return remaining_running, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))