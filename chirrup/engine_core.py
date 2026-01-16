import asyncio
import queue
import threading
from typing import List, Dict, Any, Union, Tuple, Optional, Callable, AsyncIterator
from typing_extensions import TypedDict
import uuid
import torch

from chirrup.core_structure import (
    Task,
    ModelLoadConfig,
    DEFAULT_SAMPLING_CONFIG,
    DEFAULT_STOP_TOKENS,
)
from chirrup.interface import AsyncEngineCompletion
from chirrup.worker import Worker, TRIE_TOKENIZER


class WorkerPerformanceInfo(TypedDict):
    """Worker 性能信息"""

    worker_id: str
    avg_loop_time: float
    state_size: int
    state_offset_details: Dict[str, Tuple[int, int]]
    task_details: Dict[str, int]
    max_allocated_memory_GB: float


class ThreadSafeAsyncQueue:
    """桥接线程与 asyncio 的简易队列，确保跨线程 put 安全"""

    def __init__(self, event_loop: asyncio.AbstractEventLoop, queue: Optional[asyncio.Queue] = None):
        self.event_loop = event_loop
        self.queue: asyncio.Queue = queue if queue is not None else asyncio.Queue()

    def put_nowait(self, item):
        # 在事件循环线程中执行 put，避免跨线程直接操作 asyncio.Queue
        if self.event_loop.is_closed():
            return
        try:
            self.event_loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:
            # 事件循环已关闭或未运行，直接忽略
            pass

    def empty(self) -> bool:
        return self.queue.empty()

    def get_nowait(self):
        return self.queue.get_nowait()


class AsyncEngineCore:
    """
    核心引擎类，负责处理核心功能，包括：
    - Worker 管理
    - Task 管理
    - 线程与 asyncio 间的安全通信
    """

    def __init__(self):
        self.workers: List[Worker] = []
        self.worker_threads: List[threading.Thread] = []
        self.task_queue: queue.Queue[Task] = queue.Queue()
        self.event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

        self.worker_id_set = set()

        # 跨线程事件桥接
        self.worker_event_queue: Optional[ThreadSafeAsyncQueue] = None

        # 事件循环引用
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

        # 初始化状态
        self.is_initialized = False
        self.is_shutdown = False

        # 初始化分词器
        self.tokenizer: TRIE_TOKENIZER = None

    def init(self, worker_num: int, model_config: ModelLoadConfig, batch_size: int = 32) -> asyncio.Task:
        """
        初始化 Worker，返回一个异步任务，当全部 worker 都加载成功后完成

        Args:
            worker_num: Worker 数量
            model_config: 模型配置
            batch_size: 批处理大小

        Returns:
            asyncio.Task: 当所有 worker 加载完成后完成的异步任务
        """
        if self.is_initialized:
            raise RuntimeError("Workers already initialized")

        if self.is_shutdown:
            raise RuntimeError("Engine has been shutdown")

        # 获取当前事件循环
        try:
            self.event_loop = asyncio.get_running_loop()
        except RuntimeError:
            # 如果没有运行的事件循环，创建一个新的
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)

        # 初始化跨线程队列
        self.worker_event_queue = ThreadSafeAsyncQueue(self.event_loop, asyncio.Queue(maxsize=worker_num * 100))

        self.is_initialized = True

        self.tokenizer = TRIE_TOKENIZER(model_config.vocab_path)

        # 创建异步任务来等待所有 worker 加载完成
        async def wait_for_workers_loaded():
            """等待所有 worker 加载完成的异步任务"""
            # 直接使用跨线程队列接收 worker 加载完成的消息
            if self.worker_event_queue is None:
                raise RuntimeError("worker_event_queue not initialized")

            loaded_queue = self.worker_event_queue.queue

            self.worker_id_set = {f"worker_{i}" for i in range(worker_num)}

            # 创建并启动 Worker 线程

            for k, worker_id in enumerate(self.worker_id_set):
                gpu_id = [k]  # 假设每个 Worker 使用一个 GPU

                worker = Worker(
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    model_config=model_config,
                    task_queue=self.task_queue,
                    master_event_queue=self.event_queue,
                    worker_event_queue=self.worker_event_queue,
                    batch_size=batch_size,
                    # enable_rapid_sampling=True
                )

                self.workers.append(worker)

                # 在独立线程中启动 Worker
                worker_thread = threading.Thread(target=worker.start, daemon=True, name=f"chirrup:{worker_id}")
                worker_thread.start()
                self.worker_threads.append(worker_thread)

            # 等待所有 worker 加载完成
            loaded_workers = set()
            timeout = 300  # 5分钟超时

            try:
                while len(loaded_workers) < worker_num and timeout > 0:
                    try:
                        worker_id, message_type, payload = await asyncio.wait_for(loaded_queue.get(), timeout=1.0)
                        if message_type == "worker_loaded" and payload.get("status") == "success":
                            loaded_workers.add(worker_id)
                        else:
                            print(f"Worker {worker_id} 加载失败: {payload}")
                            raise RuntimeError(f"Worker {worker_id} failed to load")
                    except asyncio.TimeoutError:
                        timeout -= 1

                if len(loaded_workers) < worker_num:
                    failed_workers = self.worker_id_set - loaded_workers
                    raise RuntimeError(f"以下 worker 加载超时: {failed_workers}")

                print(f"所有 {worker_num} 个 worker 加载完成")
            finally:
                pass

        return asyncio.create_task(wait_for_workers_loaded())

    def completion(
        self,
        prompt_str: str,
        prefill_tokens: Optional[List[int]] = None,
        state: Optional[Union[None, List[torch.Tensor]]] = None,
        priority: int = 0,
        temperature: float = DEFAULT_SAMPLING_CONFIG["temperature"],
        top_p: float = DEFAULT_SAMPLING_CONFIG["top_p"],
        top_k: int = DEFAULT_SAMPLING_CONFIG["top_k"],
        presence_penalty: float = DEFAULT_SAMPLING_CONFIG["presence_penalty"],
        frequency_penalty: float = DEFAULT_SAMPLING_CONFIG["frequency_penalty"],
        penalty_decay: float = DEFAULT_SAMPLING_CONFIG["penalty_decay"],
        stop_tokens: Optional[List[int]] = DEFAULT_STOP_TOKENS,
        forbidden_tokens: Optional[List[int]] = [],
        max_tokens: Optional[int] = DEFAULT_SAMPLING_CONFIG["max_tokens"],
        task_id: Optional[str] = None,
        cache_prefill: bool = False,
        cache_prefill_padding: int = 0,
    ) -> AsyncEngineCompletion:
        """
        创建一个 AsyncEngineCompletion 对象，并输入相应配置信息

        Args:
            prompt_str: 提示字符串
            prefill_tokens: 输入token列表
            state: 模型状态
            priority: 任务优先级
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            presence_penalty: 存在惩罚
            frequency_penalty: 频率惩罚
            penalty_decay: 惩罚衰减
            stop_tokens: 停止token列表
            forbidden_tokens: 禁用token列表
            max_tokens: 最大生成token数
            task_id: 任务ID，如果不提供则自动生成

        Returns:
            AsyncEngineCompletion 对象
        """
        assert not (
            state is not None and prefill_tokens is None
        ), "prefill_tokens cannot be None when state is not None"

        if not self.is_initialized:
            raise RuntimeError("Engine not initialized")

        if self.is_shutdown:
            raise RuntimeError("Engine has been shutdown")

        # 确保 task_id 存在
        if task_id is None:
            task_id = str(uuid.uuid4())

        if not prefill_tokens:
            prefill_tokens = self.tokenizer.encode(prompt_str)

        # 跨线程输出桥：worker 线程 put，async 端 get
        result_channel = ThreadSafeAsyncQueue(self.event_loop)

        # 创建 AsyncEngineCompletion 对象
        completion = AsyncEngineCompletion(
            prompt_str=prompt_str,
            prefill_tokens=prefill_tokens,
            state=state,
            task_queue=self.task_queue,
            priority=priority,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_decay=penalty_decay,
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
            task_id=task_id,
            result_channel=result_channel,
            forbidden_tokens=forbidden_tokens,
            cache_prefill=cache_prefill,
            cache_prefill_padding=cache_prefill_padding,
        )

        return completion

    def shutdown(self) -> None:
        """
        关闭引擎，清理资源
        """
        if self.is_shutdown:
            return

        self.is_shutdown = True

        # 发送关闭信号给所有 Worker
        shutdown_event = {"type": "shutdown"}
        try:
            self.event_queue.put_nowait(shutdown_event)
        except Exception as e:
            print(f"Failed to send shutdown signal: {e}")

        # 等待线程结束
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5)

    async def iter_worker_performance(
        self, timeout: float = 1.0
    ) -> AsyncIterator[WorkerPerformanceInfo]:
        """
        异步生成器，持续迭代获取 Worker 性能信息

        Args:
            timeout: 等待下一条性能信息的超时时间（秒）

        Yields:
            WorkerPerformanceInfo: Worker 性能信息

        Example:
            async for perf in engine.iter_worker_performance():
                print(f"Worker {perf['worker_id']}: {perf['avg_loop_time']:.4f}s/loop")
        """
        if self.worker_event_queue is None:
            raise RuntimeError("Engine not initialized")

        while not self.is_shutdown:
            try:
                message = await asyncio.wait_for(
                    self.worker_event_queue.queue.get(), timeout=timeout
                )
                worker_id, message_type, payload = message
                if message_type == "worker_performance":
                    yield WorkerPerformanceInfo(
                        worker_id=worker_id,
                        avg_loop_time=payload["avg_loop_time"],
                        state_size=payload["state_size"],
                        state_offset_details=payload["state_offset_details"],
                        task_details=payload["task_details"],
                        max_allocated_memory_GB=payload["max_allocated_memory_GB"],
                    )
            except asyncio.TimeoutError:
                continue

    def __del__(self):
        """析构函数，确保资源被清理"""
        try:
            self.shutdown()
        except Exception:
            pass  # 析构函数中忽略异常
