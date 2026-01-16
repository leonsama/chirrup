from dataclasses import dataclass, field
import enum
import asyncio
import queue
from typing import List, Optional, Union, Tuple, Dict
from typing_extensions import TypedDict
import torch

import uuid

# copy from https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/__init__.py#L24

FINISH_REASON_STRINGS = ("stop", "length", "abort")

DEFAULT_STOP_TOKENS = [0, 261, 24281]


class DEFAULT_SAMPLING_CONFIG_TYPE(TypedDict):
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    frequency_penalty: float
    penalty_decay: float
    max_tokens: int


DEFAULT_SAMPLING_CONFIG: DEFAULT_SAMPLING_CONFIG_TYPE = {
    "temperature": 1.0,
    "top_p": 0.3,
    "top_k": 0,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.5,
    "penalty_decay": 0.996,
    "max_tokens": 8192,
}


class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, or abort.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted for another reason

    """

    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


# inspired by vllm


class RequestStatus(enum.IntEnum):
    """Status of a task."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.FINISHED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:

        return _FINISHED_REASON_MAP.get(status)


_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
}


@dataclass
class Task:
    """Represents a text generation task with associated configuration and state.

    This class encapsulates all necessary information for managing a single
    inference request in an RWKV-based generation system, including input,
    decoding parameters, execution state, and output handling.

    Args:
        output_queue (asyncio.Queue[Union[Tuple[int, str], Dict]]): Output queue
            for streaming generated results. Must be provided externally.
        task_event_queue (asyncio.Queue): Event queue for task-specific events
            like abort signals.
        task_id (Optional[str]): Unique identifier for the task. If not provided,
            a UUID4 will be generated automatically.
        priority (int): Task priority level; higher values indicate higher priority.
            Defaults to 0.
        temperature (float): Sampling temperature for output randomness.
            Higher values increase diversity. Defaults to 1.0.
        top_p (float): Nucleus sampling parameter. Limits sampling to the smallest
            set of tokens with cumulative probability >= top_p. Defaults to 1.0.
        top_k (int): Top-K sampling parameter. Limits sampling to the top-K most
            probable tokens. Defaults to 0.
        presence_penalty (float): Penalty applied to tokens that have appeared
            in the output to reduce repetition. Defaults to 0.0.
        frequency_penalty (float): Penalty proportional to token appearance
            frequency in the output. Defaults to 0.0.
        penalty_decay (float): Decay factor for penalties over time.
            Should be in (0, 1]. Defaults to 0.9965.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
            Defaults to 4096.
        prompt_str (str): Formatted prompt string used as input for generation.
            Defaults to empty string.
        stop_tokens (List[int]): List of token IDs that trigger early stopping.
            Defaults to empty list.
        forbidden_tokens (List[int]): List of token IDs that are forbidden in the output.
            Defaults to empty list.
        prefill_tokens (List[int]): Tokenized input sequence for prefill.
            Defaults to empty list.
        state (Union[None,List[torch.Tensor]]): RWKV model state from prior inference.
            None if starting fresh; otherwise a list of tensors.
        cache_prefill (bool): Cache the prefill tokens.
            Defaults to False.
        cache_prefill_padding (int): Padding for cache prefill.
            Defaults to 0.
    """

    output_queue: asyncio.Queue[Union[Tuple[int, str], "Task"]]
    task_event_queue: queue.Queue  # 线程安全队列，用于 abort 等控制信号
    prompt_str: str
    prefill_tokens: List[int]
    state: Union[None, List[torch.Tensor]]
    task_id: Optional[str] = None
    priority: int = 0

    temperature: float = DEFAULT_SAMPLING_CONFIG["temperature"]
    top_p: float = DEFAULT_SAMPLING_CONFIG["top_p"]
    top_k: int = DEFAULT_SAMPLING_CONFIG["top_k"]
    presence_penalty: float = DEFAULT_SAMPLING_CONFIG["presence_penalty"]
    frequency_penalty: float = DEFAULT_SAMPLING_CONFIG["frequency_penalty"]
    penalty_decay: float = DEFAULT_SAMPLING_CONFIG["penalty_decay"]
    max_tokens: Optional[int] = DEFAULT_SAMPLING_CONFIG["max_tokens"]

    stop_tokens: List[int] = field(default_factory=lambda: DEFAULT_STOP_TOKENS)
    forbidden_tokens: List[int] = field(default_factory=list)

    cache_prefill: bool = field(default=False)
    cache_prefill_padding: int = field(default=0)

    # Internal state (not part of public API)
    event_list: List = field(init=False, default_factory=list)
    request_status: RequestStatus = field(init=False, default=RequestStatus.WAITING)
    generated_tokens: List[int] = field(init=False, default_factory=list)
    decoded_texts: List[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.task_id is None:
            self.task_id = str(uuid.uuid4())

    def is_finished(self) -> bool:
        """Check whether the task has reached a terminal state.

        Returns:
            bool: True if the request status indicates completion or failure,
            False otherwise.
        """
        return RequestStatus.is_finished(self.request_status)


@dataclass
class ModelLoadConfig:
    model_path: str
    vocab_path: str
    vocab_size: int
    head_size: int
    dtype: torch.dtype = torch.float16

    # 将由模型设定动态传入，不初始化
    n_head: Optional[int] = field(default=None, init=False)
    n_embd: Optional[int] = field(default=None, init=False)
    n_layer: Optional[int] = field(default=None, init=False)

    @property
    def param_byte(self) -> int:
        """根据 dtype 自动计算参数字长（bytes）"""
        dtype_to_bytes = {
            torch.float16: 2,
            torch.float32: 4,
            torch.bfloat16: 2,
            torch.int8: 1,
        }
        return dtype_to_bytes.get(self.dtype, 2)  # 默认 fallback 为 2

    def load_params(self, n_head: int, n_embd: int, n_layer: int) -> None:
        """从 RWKV_x070 模型设定中加载结构参数"""
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_layer = n_layer

    def get_state_size_mb(self) -> float:
        """
        计算 state 所需内存大小（单位：MB）
        state 定义如下：
            state[0] = (n_layer, 2, 1, n_embd)
            state[1] = (n_layer, 1, n_embd // head_size, head_size, head_size)
            state[2] = (vocab_size,)
        """
        if self.n_layer is None or self.n_embd is None:
            raise ValueError("n_layer 和 n_embd 必须先通过 load_params 设置")

        # state[0]
        size0 = self.n_layer * 2 * 1 * self.n_embd
        # state[1]
        size1 = self.n_layer * 1 * (self.n_embd // self.head_size) * self.head_size * self.head_size
        # state[2]
        size2 = self.vocab_size

        total_elements = size0 + size1 + size2
        total_bytes = total_elements * self.param_byte
        return total_bytes / (1024 * 1024)  # 转为 MB


if __name__ == "__main__":
    task = Task(
        asyncio.Queue(),
        prompt_str="hello world",
        prefill_tokens=[1, 2, 3, 4, 5],
        state=None,
    )
    print(task)
    print(task.is_finished())
