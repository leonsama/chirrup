import asyncio
import queue
import torch
import uuid
from typing import Optional, List, Callable, Union, Tuple, Any, Literal, TypedDict, Protocol

from chirrup.core_structure import Task, DEFAULT_SAMPLING_CONFIG, DEFAULT_STOP_TOKENS


class CachePrefill(TypedDict):
    state: list[torch.Tensor]
    prefilled_tokens: List[int]


TASK_RETURN_TYPE = Union[Tuple[Literal["token"], int, str], Tuple[Literal["cache_prefill"], CachePrefill]]


class ResultChannel(Protocol):
    """跨线程结果通道协议，Worker 端写入，Async 端读取"""

    def put_nowait(self, item: Any) -> None: ...

    @property
    def queue(self) -> asyncio.Queue: ...


class AsyncEngineCompletion:
    """异步生成任务的控制器，管理单个生成请求的生命周期"""

    def __init__(
        self,
        # 输入参数
        prompt_str: str,
        prefill_tokens: List[int],
        state: Union[None, List[torch.Tensor]],
        # 流程控制
        task_queue: queue.Queue[Task],
        result_channel: ResultChannel,
        task_id: str,
        priority: int = 0,
        # 采样参数
        temperature: float = DEFAULT_SAMPLING_CONFIG["temperature"],
        top_p: float = DEFAULT_SAMPLING_CONFIG["top_p"],
        top_k: int = DEFAULT_SAMPLING_CONFIG["top_k"],
        presence_penalty: float = DEFAULT_SAMPLING_CONFIG["presence_penalty"],
        frequency_penalty: float = DEFAULT_SAMPLING_CONFIG["frequency_penalty"],
        penalty_decay: float = DEFAULT_SAMPLING_CONFIG["penalty_decay"],
        stop_tokens: Optional[List[int]] = DEFAULT_STOP_TOKENS,
        forbidden_tokens: Optional[List[int]] = None,
        max_tokens: Optional[int] = DEFAULT_SAMPLING_CONFIG["max_tokens"],
        cache_prefill: bool = False,
        cache_prefill_padding: int = 0,
    ):
        self.task_id = task_id

        # 创建任务特定的事件队列（线程安全）
        self.task_event_queue: queue.Queue = queue.Queue()

        # result_channel: Worker 端通过 put_nowait 写入，Async 端通过 .queue 读取
        self._result_channel = result_channel
        self._result_queue = result_channel.queue  # asyncio.Queue，用于 async 读取

        self.task = Task(
            task_id=self.task_id,
            priority=priority,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_decay=penalty_decay,
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
            prompt_str=prompt_str,
            prefill_tokens=prefill_tokens,
            state=state,
            output_queue=result_channel,  # Worker 通过此队列写入结果
            task_event_queue=self.task_event_queue,
            forbidden_tokens=forbidden_tokens if forbidden_tokens is not None else [],
            cache_prefill=cache_prefill,
            cache_prefill_padding=cache_prefill_padding,
        )

        self._task_queue = task_queue

        self._submitted = False  # 任务是否已提交到队列
        self.is_finished = False

    def start(self):
        """将任务提交到队列"""
        self._submitted = True
        self._task_queue.put_nowait(self.task)

    def __aiter__(self):
        if not self._submitted:
            self.start()

        return self

    async def __anext__(
        self,
    ) -> TASK_RETURN_TYPE:
        if self.is_finished:
            raise RuntimeError("Already finished")

        while True:
            out = await self._result_queue.get()

            if isinstance(out, tuple) and len(out) == 2:
                message_type, payload = out
                if message_type == "token_generated":
                    return ("token", *payload)
                elif message_type == "task_completed":
                    self.is_finished = True
                    self.task = payload
                    raise StopAsyncIteration
                elif message_type == "cache_prefill":
                    return ("cache_prefill", payload)

            else:
                print("Unknown message format:", out)

    def get_full_completion(self) -> asyncio.Task[str]:
        """获取完整生成结果的异步任务"""
        async def fetch_all_tokens() -> str:
            result = []
            async for event in self:
                if event[0] == "token":
                    result.append(event[2])
            return "".join(result)

        return asyncio.create_task(fetch_all_tokens())

    def abort(self):
        """中止任务"""
        self.task_event_queue.put_nowait(("abort", None))
