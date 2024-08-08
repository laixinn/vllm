from vllm.entrypoints.openai.protocol import OpenAIBaseModel
from vllm.sequence import RequestMetrics
from dataclasses import dataclass

@dataclass
class MetricInfo(OpenAIBaseModel):
    first_token_time: float
    time_between_scheduled: float
    avg_itl_in_scheduled: float
    stall_count: int
    def __init__(self, metrics: RequestMetrics):
        assert len(metrics.every_enque) == len(metrics.every_deque)
        assert len(metrics.every_enque) - metrics.decode_index == len(metrics.every_token_num)
        self.first_token_time: float = metrics.time_in_queue
        self.time_between_scheduled: float = sum([
            e-d for e,d in 
                zip(metrics.every_enque[1:], metrics.every_deque[:-1])
        ])
        self.avg_itl_in_scheduled: float = sum([
            d-e for e,d in 
                zip(metrics.every_enque[metrics.decode_index], metrics.every_deque[metrics.decode_index])
        ]) / sum(metrics.every_token_num)
        self.stall_count: int = sum([
            e-d > 0.3 for e,d in
                zip(metrics.every_enque[metrics.decode_index][1:], metrics.every_deque[metrics.decode_index][:-1])
        ])