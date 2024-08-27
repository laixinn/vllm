import asyncio
import time
from functools import partial
from typing import (AsyncIterator, Callable, Dict, Iterable, List, Optional,
                    Set, Tuple, Type, Union)
from collections import deque

from transformers import PreTrainedTokenizer

import vllm.envs as envs
from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.executor.ray_utils import initialize_ray_cluster, ray
from vllm.inputs import LLMInputs, PromptInputs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.usage.usage_lib import UsageContext

from threading import Event, Thread, Lock
from queue import Queue
from typing import Sequence as GenericSequence
from vllm.sequence import (PoolerOutput, SequenceGroup, SequenceGroupMetadata)
from vllm.core.scheduler import ScheduledSequenceGroup
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.outputs import RequestOutputFactory

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S
REQUEST_LOOP_DELAY = 0.01
STEP_LOOP_DELAY = 0.01

class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")

    exception = None
    try:
        task.result()
        # NOTE: This will be thrown if task exits normally (which it should not)
        raise AsyncEngineDeadError(msg)
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            msg + " See stack trace above for the actual cause.") from e


class AsyncStream:
    """A stream of RequestOutputs or EmbeddingRequestOutputs for a request
    that can be iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, EmbeddingRequestOutput,
                              Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> Union[RequestOutput, EmbeddingRequestOutput]:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(request_id)
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(rid)

    def process_request_output(self,
                               request_output: Union[RequestOutput,
                                                     EmbeddingRequestOutput],
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info("Finished request %s.", request_id)
            self.abort_request(request_id)

    def process_exception(self,
                          request_id: str,
                          exception: Exception,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info("Finished request %s.", request_id)
        self.abort_request(request_id)

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info("Aborted request %s.", request_id)

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.schedule_consumption = deque([], maxlen=100)
        self.forward_consumption = deque([], maxlen=100)
        self.output_consumption = deque([], maxlen=100)
        self.pipeline_consumption = deque([], maxlen=100)
        self.schedule_cache_consumption = deque([], maxlen=100)
        self.pipeline_loss = deque([], maxlen=100)
        self.pipeline_event_loss = deque([], maxlen=100)
        self.pipeline_schedule_loss = deque([], maxlen=100)

    async def step_async(
            self) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        t0 = time.time()
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        t1 = time.time()
        self.schedule_consumption.append(t1 - t0)

        if not scheduler_outputs.is_empty():
            # Execute the model.
            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
            )
            output = await self.model_executor.execute_model_async(
                execute_model_req)
        else:
            output = []
        t2 = time.time()
        self.forward_consumption.append(t2 - t1)

        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)
        t3 = time.time()
        self.output_consumption.append(t3 - t2)

        # # Log average consumption
        # logger.info(f"Avg schedule time: {sum(self.schedule_consumption) / len(self.schedule_consumption)};" + 
        #             f" Avg forward time: {sum(self.forward_consumption) / len(self.forward_consumption)};" + 
        #             f" Avg output time: {sum(self.output_consumption) / len(self.output_consumption)}")

        # for res_output in request_outputs:
        #     logger.info(f"Request {res_output.request_id} is scheduled, new text:" +
        #                 ";".join([output.text for output in res_output.outputs]))

        # Log stats.
        self.do_log_stats(scheduler_outputs, output)

        if not request_outputs:
            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            await self.model_executor.stop_remote_worker_execution_loop_async()

        return request_outputs

    async def process_model_inputs_async(
        self,
        request_id: str,
        inputs: PromptInputs,
        lora_request: Optional[LoRARequest] = None,
    ) -> LLMInputs:
        if isinstance(inputs, str):
            inputs = {"prompt": inputs}

        if "prompt_token_ids" not in inputs:
            tokenizer = self.get_tokenizer_group("prompts must be None if "
                                                 "skip_tokenizer_init is True")

            prompt_token_ids = await tokenizer.encode_async(
                request_id=request_id,
                prompt=inputs["prompt"],
                lora_request=lora_request)
        else:
            prompt_token_ids = inputs["prompt_token_ids"]

        return LLMInputs(prompt_token_ids=prompt_token_ids,
                         prompt=inputs.get("prompt"),
                         multi_modal_data=inputs.get("multi_modal_data"))

    async def add_request_async(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()

        processed_inputs = await self.process_model_inputs_async(
            request_id=request_id, inputs=inputs, lora_request=lora_request)

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
        )

    async def check_health_async(self) -> None:
        self.model_executor.check_health()

class _AsyncPipelineLLMEngineV1(_AsyncLLMEngine):
    """Extension of LLMEngine to add async methods."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # pipeline
        self.schedule_event = Event()
        self.schedule_thread = Thread(target=self._schedule_thread, daemon=True)
        self.schedule_output = deque()

        self.forward_event = Event()
        self.forward_thread = Thread(target=self._forward_thread, daemon=True)
        self.forward_output = deque()

        self.stream_event = Event()
        self.stream_thread = Thread(target=self._stream_thread, daemon=True)
        self.stream_output = deque()
        self.has_stream_output = Event()

    def _schedule_thread(self):
        while True:
            self.schedule_event.wait()
            self.schedule_event.clear()

            t0 = time.time()
            self.scheduler.cache_consumption = 0
            seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
            
            t1 = time.time()
            self.schedule_consumption.append(t1 - t0)
            self.schedule_cache_consumption.append(self.scheduler.cache_consumption)

            self.schedule_output.append([seq_group_metadata_list, scheduler_outputs])
            self.forward_event.set()
    
    def _forward_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                self.forward_event.wait()
                self.forward_event.clear()

                t1 = time.time()
                assert len(self.schedule_output) > 0
                seq_group_metadata_list, scheduler_outputs = self.schedule_output.popleft()

                if not scheduler_outputs.is_empty():
                    # Execute the model.
                    execute_model_req = ExecuteModelRequest(
                        seq_group_metadata_list=seq_group_metadata_list,
                        blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                        blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                        blocks_to_copy=scheduler_outputs.blocks_to_copy,
                        num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                        running_queue_size=scheduler_outputs.running_queue_size,
                    )

                    future = loop.create_task(self.model_executor.execute_model_async(execute_model_req))
                    loop.run_until_complete(future)
                    output = future.result()
                    # output = await self.model_executor.execute_model_async(execute_model_req)
                else:
                    output = []
                t2 = time.time()
                self.forward_consumption.append(t2 - t1)

                self.forward_output.append([output, seq_group_metadata_list, scheduler_outputs])
                self.stream_event.set()
        except Exception as e:
            logger.error(f"Error in forward thread: {e}")
        finally:
            loop.close()

    def _stream_thread(self):
        while True:
            self.stream_event.wait()
            self.stream_event.clear()

            t2 = time.time()
            output, seq_group_metadata_list, scheduler_outputs = self.forward_output.popleft()

            request_outputs = self._process_model_outputs(
                output, scheduler_outputs.scheduled_seq_groups,
                scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)
            t3 = time.time()
            self.output_consumption.append(t3 - t2)

            # for res_output in request_outputs:
            #     logger.info(f"Request {res_output.request_id} is scheduled, new text:" +
            #                 ";".join([output.text for output in res_output.outputs]))

            # Log stats.
            self.do_log_stats(scheduler_outputs, output)

            if not request_outputs:
                # Stop the execute model loop in parallel workers until there are
                # more requests to process. This avoids waiting indefinitely in
                # torch.distributed ops which may otherwise timeout, and unblocks
                # the RPC thread in the workers so that they can process any other
                # queued control plane messages, such as add/remove lora adapters.
                asyncio.run(self.model_executor.stop_remote_worker_execution_loop_async())
                # await self.model_executor.stop_remote_worker_execution_loop_async()

            self.stream_output.append(request_outputs)
            self.has_stream_output.set()

    async def step_async(
            self) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        t0 = time.time()
        self.schedule_event.set()
        self.has_stream_output.wait()
        self.has_stream_output.clear()
        t1 = time.time()
        self.pipeline_consumption.append(t1 - t0)
        return self.stream_output.popleft()
    
class _AsyncPipelineLLMEngine(_AsyncLLMEngine):
    """Extension of LLMEngine for lazy scheduler."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # pipeline
        self.schedule_event = Event()
        self.schedule_thread = Thread(target=self._schedule_thread, daemon=True)
        # self.schedule_output = Queue(maxsize=1)

        self.SCHEDULER_LOOP_DELAY = 0.01
        self.schedule_timer_thread = Thread(target=self._scheduler_timer_loop, daemon=True)

        self.forward_event = Event()
        self.forward_thread = Thread(target=self._forward_thread, daemon=True)
        self.forward_output = Queue()

        # self.stream_event = Event()
        self.stream_thread = Thread(target=self._stream_thread, daemon=True)
        self.stream_output = Queue()
        # self.has_stream_output = Event()

        self.cache_lock = Lock()

    def init_threads(self) ->None:
        # self.schedule_event.set()
        self.schedule_thread.start()
        self.forward_thread.start()
        self.stream_thread.start()
        self.schedule_timer_thread.start()
    
    def _scheduler_timer_loop(self) -> None:
        while True:
            time.sleep(self.SCHEDULER_LOOP_DELAY)
            num_request = self.scheduler.get_num_running_requests()
            if num_request > 0:
                # logger.info(f"We have {num_request} requests")
                self.schedule_event.set()

    def _schedule_thread(self):
        last_run = None
        while True:
            t0 = time.time()
            self.schedule_event.wait()
            self.schedule_event.clear()

            t1 = time.time()
            self.pipeline_schedule_loss.append(t1 - t0)

            # t0 = time.time()
            # seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
            # logger.info("Scheduler is scheduling")
            scheduling_time = self.scheduler.lazy_schedule()
            self.schedule_consumption.append(scheduling_time)
            
            # t1 = time.time()
            # if last_run is None:
            #     last_run = t1
            # else:
            #     self.pipeline_schedule_loss.append(t1 - last_run - scheduling_time)
            #     last_run = t1

            # self.schedule_output.append([seq_group_metadata_list, scheduler_outputs])
            # self.forward_event.set()
    
    def _forward_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        last_run = None
        try:
            while True:
                t0 = time.time()
                # self.forward_event.wait()
                # self.forward_event.clear()

                # t1 = time.time()
                # self.pipeline_event_loss.append(t1-t0)
                # assert len(self.schedule_output) > 0
                # seq_group_metadata_list, scheduler_outputs = self.schedule_output.popleft()

                seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
                # logger.info("Got scheduler exectuor result.")
                self.schedule_cache_consumption.append(self.scheduler.cache_consumption)
                logger.info("Forward is running")

                t1 = time.time()
                self.pipeline_event_loss.append(t1 - t0 - self.scheduler.cache_consumption)

                if not scheduler_outputs.is_empty():
                    # Execute the model.
                    execute_model_req = ExecuteModelRequest(
                        seq_group_metadata_list=seq_group_metadata_list,
                        blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                        blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                        blocks_to_copy=scheduler_outputs.blocks_to_copy,
                        num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                        running_queue_size=scheduler_outputs.running_queue_size,
                    )

                    # future = loop.create_task(self.model_executor.execute_model_async(execute_model_req))
                    # loop.run_until_complete(future)
                    # output = future.result()

                    output = self.model_executor.execute_model(execute_model_req)

                    # logger.info("Got model result.")
                    # output = await self.model_executor.execute_model_async(execute_model_req)
                else:
                    output = []
                    logger.warning("Put empty output into forward output.")
                t2 = time.time()
                self.forward_consumption.append(t2 - t1 + self.scheduler.cache_consumption)

                if last_run is None:
                    last_run = t2
                else:
                    self.pipeline_loss.append(t2 - last_run - (t2-t1) - self.scheduler.cache_consumption)
                    last_run = t2

                # self.forward_output.append([output, seq_group_metadata_list, scheduler_outputs])
                self.forward_output.put([output, seq_group_metadata_list, scheduler_outputs])
                # self.stream_event.set()

                # self.schedule_event.set()
        # except Exception as e:
        #     logger.error(f"Error in forward thread: {e}")
        finally:
            loop.close()

    def _stream_thread(self):
        while True:
            # self.stream_event.wait()
            # self.stream_event.clear()

            # output, seq_group_metadata_list, scheduler_outputs = self.forward_output.popleft()
            output, seq_group_metadata_list, scheduler_outputs = self.forward_output.get()
            if len(output) == 0:
                logger.warning("Got emtpy forward output.")
                # self.scheduler.sync_running()
                continue
            logger.info("Streaming is running")

            t2 = time.time()
            free_gpu = self.scheduler.block_manager.gpu_allocator.get_num_free_blocks()
            request_outputs = self._process_model_outputs(
                output, scheduler_outputs.scheduled_seq_groups,
                scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)
            free_gpu2 = self.scheduler.block_manager.gpu_allocator.get_num_free_blocks()
            if free_gpu2 > free_gpu:
                self.scheduler.atomic.acquire()
                # logger.info(f"Free GPU: {free_gpu} -> {free_gpu2}")
                self.scheduler.free_gpu_blocks += (free_gpu2 - free_gpu)
                self.scheduler.atomic.release()
            # logger.info("Normal streaming put one token")
            # self.scheduler.sync_running()
            self.scheduler.check_sync()
            t3 = time.time()
            self.output_consumption.append(t3 - t2)

            # for res_output in request_outputs:
            #     logger.info(f"Request {res_output.request_id} is scheduled, new text:" +
            #                 ";".join([output.text for output in res_output.outputs]))

            # Log stats.
            self.do_log_stats(scheduler_outputs, output)

            if not request_outputs:
                # Stop the execute model loop in parallel workers until there are
                # more requests to process. This avoids waiting indefinitely in
                # torch.distributed ops which may otherwise timeout, and unblocks
                # the RPC thread in the workers so that they can process any other
                # queued control plane messages, such as add/remove lora adapters.
                asyncio.run(self.model_executor.stop_remote_worker_execution_loop_async())
                # await self.model_executor.stop_remote_worker_execution_loop_async()

            # self.stream_output.append(request_outputs)
            # self.has_stream_output.set()
            self.stream_output.put(request_outputs)

            # self.schedule_event.set()

    def step(
        self
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # t0 = time.time()

        # try:
        #     # logger.info(f"Stream output size: {self.stream_output.qsize()}")
        #     output = self.stream_output.get(timeout=1)
        #     # logger.info(f"Got one stream output with size {len(output)}")
        # except Exception as e:
        #     logger.warning("step timeout, put a token to scheduler")
        #     self.scheduler.sync_running()
        #     output = []

        try:
            output = self.stream_output.get_nowait()
        except Exception as e:
            logger.warning("step timeout, put a token to scheduler")
            # self.scheduler.sync_running()
            output = []

        # t1 = time.time()
        # self.pipeline_consumption.append(t1 - t0)
        return output

class AsyncLLMEngine:
    """An asynchronous wrapper for :class:`LLMEngine`.

    This class is used to wrap the :class:`LLMEngine` class to make it
    asynchronous. It uses asyncio to create a background loop that keeps
    processing incoming requests. The :class:`LLMEngine` is kicked by the
    generate method when there are requests in the waiting queue. The generate
    method yields the outputs from the :class:`LLMEngine` to the caller.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        max_log_len: Maximum number of prompt characters or prompt ID numbers
            being printed in log.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for :class:`LLMEngine`.
        **kwargs: Arguments for :class:`LLMEngine`.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine
    # _engine_class: Type[_AsyncPipelineLLMEngine] = _AsyncPipelineLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop: Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self.start_engine_loop = start_engine_loop
        self._errored_with: Optional[BaseException] = None

        # Lazy initialized fields
        self._request_tracker: RequestTracker

        self.last_time = None

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        distributed_executor_backend = (
            engine_config.parallel_config.distributed_executor_backend)

        if engine_config.device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutorAsync
            executor_class = NeuronExecutorAsync
        elif engine_config.device_config.device_type == "cpu":
            assert distributed_executor_backend is None, (
                "Distributed execution is not supported with the CPU backend.")
            from vllm.executor.cpu_executor import CPUExecutorAsync
            executor_class = CPUExecutorAsync
        elif distributed_executor_backend == "ray":
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            executor_class = RayGPUExecutorAsync
        elif distributed_executor_backend == "mp":
            from vllm.executor.multiproc_gpu_executor import (
                MultiprocessingGPUExecutorAsync)
            executor_class = MultiprocessingGPUExecutorAsync
        else:
            from vllm.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync

        '''lazy scheduler needs pipeline engine'''
        if engine_config.scheduler_config.scheduler_priority == 'lazy':
            cls._engine_class = _AsyncPipelineLLMEngine

        # Create the async LLM engine.
        engine = cls(
            distributed_executor_backend == "ray",
            engine_args.engine_use_ray,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
        )
        return engine

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())

    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None and
                                self._background_loop_unshielded is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    async def get_tokenizer(self) -> "PreTrainedTokenizer":
        if self.engine_use_ray:
            return await self.engine.get_tokenizer.remote()  # type: ignore
        else:
            return self.engine.get_tokenizer()

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()

        if self._engine_class == _AsyncPipelineLLMEngine:
            self.engine.init_threads()
            self.is_pipeline = True
            self.step_thread = Thread(target=self.engine_step_thread, daemon=True)
            self.step_thread.start()
            self._background_loop_unshielded = asyncio.get_event_loop(
            ).create_task(self.engine_request_thread())

            # self._background_loop_unshielded = asyncio.get_event_loop(
            # ).create_task(self.run_engine_loop())

            logger.info("Pipeline engine started.")
        else:
            self.is_pipeline = False
            self._background_loop_unshielded = asyncio.get_event_loop(
            ).create_task(self.run_engine_loop())

        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = kwargs["cache_config"]
            parallel_config = kwargs["parallel_config"]
            if parallel_config.tensor_parallel_size == 1:
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(
                self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            try:
                if self.engine_use_ray:
                    await self.engine.add_request.remote(  # type: ignore
                        **new_request)
                else:
                    await self.engine.add_request_async(**new_request)
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        # if len(new_requests) > 0:
            # logger.info("Add new requests.")

        if len(finished_requests) > 0:
            # logger.info("Aborting requests.")
            await self._engine_abort(finished_requests)

        if self.is_pipeline:
            request_outputs = self.engine.step()
        elif self.engine_use_ray:
            request_outputs = await self.engine.step.remote()  # type: ignore
        else:
            request_outputs = await self.engine.step_async()

        # logger.info(f"Got one stream output with size {len(request_outputs)}.")
        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)#verbose=True)#
        
        cur_time = time.time()
        if self.last_time is None:
            self.last_time = cur_time
        else:
            self.engine.pipeline_consumption.append(cur_time - self.last_time)
            self.last_time = cur_time

        return len(request_outputs) > 0
        # return False

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)  # type: ignore
        else:
            self.engine.abort_request(request_ids)

    async def engine_request_thread(self):
        while True:
            has_unfinished_requests = (
                self.engine.scheduler.get_num_running_requests() > 0 or 
                not self.engine.stream_output.empty()
            )
            if not has_unfinished_requests:
                logger.debug("Waiting for new requests...")
                self.last_time = None
                await self._request_tracker.wait_for_new_requests()
                logger.debug("Got new requests!")
                self.last_time = time.time()

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            new_requests, finished_requests = (
                self._request_tracker.get_new_and_finished_requests())

            for new_request in new_requests:
                # Add the request into the vLLM engine's waiting queue.
                # TODO: Maybe add add_request_batch to reduce Ray overhead
                try:
                    if self.engine_use_ray:
                        await self.engine.add_request.remote(  # type: ignore
                            **new_request)
                    else:
                        await self.engine.add_request_async(**new_request)
                except ValueError as e:
                    # TODO: use a vLLM specific error for failed validation
                    self._request_tracker.process_exception(
                        new_request["request_id"],
                        e,
                        verbose=self.log_requests,
                    )

            # if len(new_requests) > 0:
                # logger.info("Add new requests.")

            if len(finished_requests) > 0:
                # logger.info("Aborting requests.")
                await self._engine_abort(finished_requests)

            await asyncio.sleep(0.2)

    def engine_step_thread(self):
        # self.finished_requests = set()
        while True:
            # has_unfinished_requests = (
            #         self.engine.scheduler.get_num_running_requests() > 0 or 
            #         not self.engine.stream_output.empty()
            #     )
            has_unfinished_requests = not self.engine.stream_output.empty()
            if has_unfinished_requests:
                # logger.info(f"Starting engine step with {self.engine.scheduler.get_num_running_requests()} requests")
                request_outputs = self.engine.step()
                for request_output in request_outputs:
                    self._request_tracker.process_request_output(
                        request_output, verbose=self.log_requests)
                    
                cur_time = time.time()
                if self.last_time is None:
                    self.last_time = cur_time
                else:
                    self.engine.pipeline_consumption.append(cur_time - self.last_time)
                    self.last_time = cur_time

            time.sleep(STEP_LOOP_DELAY)

    async def run_engine_loop(self):
        has_requests_in_progress = False
        while True:
            if not self.is_pipeline:
                has_unfinished_requests = self.engine.scheduler.get_num_running_requests() > 0
            else:
                has_unfinished_requests = (
                    self.engine.scheduler.get_num_running_requests() > 0 or 
                    not self.engine.stream_output.empty()
                )
            if not has_requests_in_progress and not has_unfinished_requests:
                logger.debug("Waiting for new requests...")
                self.last_time = None
                await self._request_tracker.wait_for_new_requests()
                logger.debug("Got new requests!")

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                # logger.info(f"Starting engine step with {len(self._request_tracker._request_streams)} requests")
                has_requests_in_progress = await asyncio.wait_for(
                    self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)
                logger.debug("Finished engine step.")
            except Exception as exc:
                logger.warning("engine step time out.")
            # except asyncio.TimeoutError as exc:
            #     # logger.error(
            #     #     "Engine iteration timed out. This should never happen!")
            #     # self.set_errored(exc)
            #     # raise
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncStream:
        if self.log_requests:
            if isinstance(inputs, str):
                shortened_prompt = inputs
                shortened_token_ids = None
            else:
                shortened_prompt = inputs.get("prompt")
                shortened_token_ids = inputs.get("prompt_token_ids")

            max_log_len = self.max_log_len
            if max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:max_log_len]

            logger.info(
                "Received request %s: prompt: %r, "
                "params: %s, prompt_token_ids: %s, "
                "lora_request: %s.", request_id, shortened_prompt, params,
                shortened_token_ids, lora_request)

        if not self.is_running:
            if self.start_engine_loop:
                logger.debug("now starting background loop")
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        if arrival_time is None:
            arrival_time = time.time()

        if self.engine_use_ray:
            processed_inputs = await self.engine.process_model_inputs_async \
                .remote(  # type: ignore
                    request_id=request_id,
                    inputs=inputs,
                    lora_request=lora_request)
        else:
            processed_inputs = await self.engine.process_model_inputs_async(
                request_id=request_id,
                inputs=inputs,
                lora_request=lora_request)

        stream = self._request_tracker.add_request(
            request_id,
            inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
        )

        return stream

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            inputs: The inputs to the LLM. See
                :class:`~vllm.inputs.PromptInputs`
                for more details about the format of each input.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.

        Yields:
            The output `RequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        async for output in self._process_request(
                request_id,
                inputs,
                sampling_params,
                lora_request=lora_request,
        ):
            yield LLMEngine.validate_output(output, RequestOutput)

    async def encode(
        self,
        inputs: PromptInputs,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncIterator[EmbeddingRequestOutput]:
        """Generate outputs for a request from an embedding model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            inputs: The inputs to the LLM. See
                :class:`~vllm.inputs.PromptInputs`
                for more details about the format of each input.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.

        Yields:
            The output `EmbeddingRequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "input": "What is LLM?",
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.encode(
            >>>    example_input["input"],
            >>>    PoolingParams(),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        async for output in self._process_request(
                request_id,
                inputs,
                pooling_params,
                lora_request=lora_request,
        ):
            yield LLMEngine.validate_output(output, EmbeddingRequestOutput)

    async def _process_request(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Union[SamplingParams, PoolingParams],
        *,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncIterator[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Common logic to process requests with SamplingParams or
        PoolingParams."""
        arrival_time = time.time()

        stream = await self.add_request(
            request_id,
            inputs,
            params,
            arrival_time=arrival_time,
            lora_request=lora_request,
        )

        try:
            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()  # type: ignore
        else:
            return self.engine.get_model_config()

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_decoding_config.remote(  # type: ignore
            )
        else:
            return self.engine.get_decoding_config()

    async def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        if self.engine_use_ray:
            await self.engine.do_log_stats.remote(  # type: ignore
                scheduler_outputs, model_output)
        else:
            self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        if self.engine_use_ray:
            try:
                await self.engine.check_health.remote()  # type: ignore
            except ray.exceptions.RayActorError as e:
                raise RuntimeError("Engine is dead.") from e
        else:
            await self.engine.check_health_async()
        logger.debug("Health check took %fs", time.perf_counter() - t)
