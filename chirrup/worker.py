import asyncio
import queue
import gc
import types
import time
import threading
from typing import List, Dict, Optional, Any, Tuple
import torch
from collections import deque

from chirrup.core_structure import Task, ModelLoadConfig, RequestStatus, FinishReason
from chirrup.utils.samplers import sample_logits_real_batch
from chirrup.utils.rapid_sampling_wrapper import load_rapid_sampling

# 定义TaskData的类型结构
from typing_extensions import TypedDict

from Albatross.rwkv7 import RWKV_x070 as RWKV_x070_ORIGINAL
from Albatross.utils import TRIE_TOKENIZER

from collections import defaultdict

from enum import IntEnum, auto


# For 3.14
def ensure_instance_annotations(cls):
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "__annotations__"):
            self.__annotations__ = {}
        if init is not object.__init__:
            init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


RWKV_x070 = ensure_instance_annotations(RWKV_x070_ORIGINAL)


def min_swaps_to_target_fast(lst, elements: list[int]):
    swaps: List[Tuple[int, int]] = []
    target = sorted(lst)

    # 构建每个字符在 target 中的位置队列
    pos_map = defaultdict(list)
    for idx, val in enumerate(lst):
        pos_map[val].append(idx)

    offsets: List[Tuple[int, int]] = []
    offset = 0

    for target in elements:
        if target not in pos_map:
            offsets.append((offset, offset))
            continue

        pos = pos_map[target]
        target_count = len(pos)

        offsets.append((offset, offset + target_count))

        target_should_move_back_id = [i for i in pos if i >= target_count + offset]
        target_avaliable_id = [i for i in range(offset, target_count + offset) if i not in pos]

        for k, v in enumerate(target_should_move_back_id):
            swap = (target_avaliable_id[k], v)
            swaps.append((target_avaliable_id[k], v))
            lst[swap[0]], lst[swap[1]] = lst[swap[1]], lst[swap[0]]

        offset += target_count
        pos_map = defaultdict(list)
        for idx, val in enumerate(lst[offset:]):
            pos_map[val].append(idx + offset)

    return swaps, offsets


# 全局模型编译锁，防止多线程同时编译 Torch JIT 模型
_MODEL_COMPILE_LOCK = threading.Lock()
_MODEL_COMPILE_DONE = False


class StateCategory(IntEnum):
    FORWARD_ONE = auto()
    FORWARD_ONE_SUSPENDED = auto()
    FORWARD_SEQ = auto()
    FINISHED = auto()
    EMPTY = auto()


class TaskData(TypedDict):
    task: Optional[Task]
    # state_pos: int
    new_token: Optional[int]
    next_input_token: Optional[int]
    is_prefilling: Optional[bool]
    state_category: StateCategory
    prefilled_tokens: List[int]
    prefill_cached: bool


class Worker:
    """
    Worker 类用于处理 Task，实现 continuous batching。

    Worker 在独立线程中运行，维护一个任务池并执行模型推理。
    """

    def __init__(
        self,
        worker_id: str,
        gpu_id: List[int],
        model_config: ModelLoadConfig,
        task_queue: queue.Queue,
        master_event_queue: queue.Queue,
        worker_event_queue: queue.Queue,
        batch_size: int = 32,
        enable_rapid_sampling: bool = False,
    ):
        """
        初始化 Worker

        Args:
            gpu_id: 分配给 Worker 的 GPU ID 列表
            model_config: 模型加载配置
            task_queue: 任务队列，Worker 消费该队列
            master_event_queue: 事件队列，包含调度要求
            batch_size: 批处理大小
            enable_rapid_sampling: 是否启用 Rapid-Sampling 采样
        """
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model_config = model_config
        self.task_queue = task_queue
        self.master_event_queue = master_event_queue
        self.worker_event_queue = worker_event_queue

        self.real_state_size = batch_size
        self.max_batch_size = batch_size - 1
        self.max_prefill_count = max(int(batch_size * 0.125), 1)

        # Worker 内部数据
        # self.task_pool: List[TaskData] = []
        self.state_slot: dict[int, TaskData] = {
            i: {
                "task": None,
                "is_prefilling": None,
                "new_token": None,
                "next_input_token": None,
                "state_category": StateCategory.EMPTY,
            }
            for i in range(self.max_batch_size)
        }

        self.model: RWKV_x070_ORIGINAL = None

        self.batch_state: list[torch.Tensor] = None
        self.occurrence: torch.Tensor = None
        self.alpha_presence_vector: torch.Tensor = None

        # 解码参数预处理 tensor
        self.temperature_tensor: torch.Tensor = None
        self.top_p_tensor: torch.Tensor = None
        self.top_k_tensor: torch.Tensor = None
        self.frequency_penalty_tensor: torch.Tensor = None
        self.penalty_decay_tensor: torch.Tensor = None
        self.presence_penalty_tensor: torch.Tensor = None

        self.no_penalty_token_ids = {33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58}

        # Rapid-Sampling 相关
        self.enable_rapid_sampling = enable_rapid_sampling
        self.rapid_sampling_module = None
        self.rapid_sampling_states: torch.Tensor = None
        self.penalties: torch.Tensor = None  # rapid-sampling 使用的 penalties tensor

        # seq foward
        self.min_forward_seq_len = 10
        self.max_forward_seq_len_per_forward = 100
        self.seq_forward_count_down = 0
        self.decode_prefill_ratio: int = 5

        self.shutdown_flag = False
        self.tokenizer: TRIE_TOKENIZER = None

        self.loop_time_recorder = deque(maxlen=10)

    def _send_worker_loaded_message(self):
        """发送 Worker 加载成功信息"""
        try:
            message = (
                self.worker_id,
                "worker_loaded",
                {
                    "status": "success",
                    "worker_id": self.worker_id,
                    "gpu_id": self.gpu_id,
                    "model_path": self.model_config.model_path,
                },
            )
            self.worker_event_queue.put_nowait(message)
            print(f"[{self.worker_id}] 发送加载成功信息")
        except Exception as e:
            print(f"[{self.worker_id}] 发送加载信息失败: {e}")

    def _load_model(self):
        """加载模型"""
        global _MODEL_COMPILE_LOCK, _MODEL_COMPILE_DONE

        # 使用全局锁确保同一时间只有一个线程在进行模型编译
        with _MODEL_COMPILE_LOCK:
            # 如果模型已经编译完成，直接创建新实例
            if _MODEL_COMPILE_DONE:
                args = types.SimpleNamespace()
                args.vocab_size = self.model_config.vocab_size
                args.head_size = self.model_config.head_size
                if self.model_config.model_path.endswith(".pth"):
                    args.MODEL_NAME = self.model_config.model_path[:-4]
                else:
                    args.MODEL_NAME = self.model_config.model_path

                self.model = RWKV_x070(args)
                self.tokenizer = TRIE_TOKENIZER(self.model_config.vocab_path)

                # 发送成功加载信息
                self._send_worker_loaded_message()
                return

            # 第一次加载模型，需要进行 JIT 编译
            print(f"[{self.worker_id}] 开始编译模型，请稍候...")
            try:
                args = types.SimpleNamespace()
                args.vocab_size = self.model_config.vocab_size
                args.head_size = self.model_config.head_size
                if self.model_config.model_path.endswith(".pth"):
                    args.MODEL_NAME = self.model_config.model_path[:-4]
                else:
                    args.MODEL_NAME = self.model_config.model_path

                self.model = RWKV_x070(args)
                self.tokenizer = TRIE_TOKENIZER(self.model_config.vocab_path)

                # 标记编译完成
                _MODEL_COMPILE_DONE = True
                print(f"[{self.worker_id}] 模型编译完成")

                # 发送成功加载信息
                self._send_worker_loaded_message()

            except Exception as e:
                print(f"[{self.worker_id}] 模型编译失败: {e}")
                raise

    def _init_worker(self):

        # 设置 GPU
        if self.gpu_id:
            torch.cuda.set_device(self.gpu_id[0])
        self._load_model()

        # 预分配
        self.batch_state = self.model.generate_zero_state(self.real_state_size)

        # Rapid-Sampling 初始化
        if self.enable_rapid_sampling:
            self.rapid_sampling_module = load_rapid_sampling()
            self.rapid_sampling_states = self.rapid_sampling_module.setup_rand(42, self.real_state_size)
            self.penalties = torch.zeros(
                (self.real_state_size, self.model_config.vocab_size),
                dtype=torch.float32,
                device=self.batch_state[0].device,
            )
            print(f"[{self.worker_id}] Rapid-Sampling 已启用")
        else:
            self.occurrence = torch.zeros(
                (self.real_state_size, self.model_config.vocab_size),
                dtype=torch.float32,
                device=self.batch_state[0].device,
            )
            self.alpha_presence_vector = torch.zeros(
                (self.real_state_size, self.model_config.vocab_size),
                dtype=torch.float32,
                device=self.batch_state[0].device,
            )
            self.temperature_tensor = torch.zeros(
                (self.real_state_size, 1),
                dtype=torch.float16,
                device=self.batch_state[0].device,
            )
            self.top_p_tensor = torch.zeros(
                (self.real_state_size, 1),
                dtype=torch.float16,
                device=self.batch_state[0].device,
            )
            self.top_k_tensor = torch.zeros(
                (self.real_state_size, 1),
                dtype=torch.int32,
                device=self.batch_state[0].device,
            )
            self.frequency_penalty_tensor = torch.zeros(
                (self.real_state_size, 1),
                dtype=torch.float16,
                device=self.batch_state[0].device,
            )
            self.penalty_decay_tensor = torch.zeros(
                (self.real_state_size, 1),
                dtype=torch.float16,
                device=self.batch_state[0].device,
            )
            self.presence_penalty_tensor = torch.zeros(
                (self.real_state_size, 1),
                dtype=torch.float32,
                device=self.batch_state[0].device,
            )

            print(f"[{self.worker_id}] Rapid-Sampling 未启用")


    def _switch_batch(self, pos_a: int, pos_b: int):
        if pos_a == pos_b:
            return

        assert (
            pos_a < self.max_batch_size and pos_b < self.max_batch_size
        ), f"pos_a {pos_a}, pos_b {pos_b}, max_batch_size {self.max_batch_size}, real_state_size {self.real_state_size}; pos_a and pos_b shall be less than max_batch_size."

        # switch state

        # cache pos_a
        self.batch_state[0][:, :, [self.real_state_size - 1], :] = self.batch_state[0][:, :, [pos_a], :]
        self.batch_state[1][:, [self.real_state_size - 1], :, :] = self.batch_state[1][:, [pos_a], :, :]
        self.batch_state[2][[self.real_state_size - 1]] = self.batch_state[2][[pos_a]]

        # pos_b -> pos_a
        self.batch_state[0][:, :, [pos_a], :] = self.batch_state[0][:, :, [pos_b], :]
        self.batch_state[1][:, [pos_a], :, :] = self.batch_state[1][:, [pos_b], :, :]
        self.batch_state[2][[pos_a]] = self.batch_state[2][[pos_b]]

        # cached pos_a -> pos_b
        self.batch_state[0][:, :, [pos_b], :] = self.batch_state[0][:, :, [self.real_state_size - 1], :]
        self.batch_state[1][:, [pos_b], :, :] = self.batch_state[1][:, [self.real_state_size - 1], :, :]
        self.batch_state[2][[pos_b]] = self.batch_state[2][[self.real_state_size - 1]]

        # Rapid-Sampling penalties
        if self.enable_rapid_sampling:
            self.penalties[[self.real_state_size - 1], :] = self.penalties[[pos_a], :]
            self.penalties[[pos_a], :] = self.penalties[[pos_b], :]
            self.penalties[[pos_b], :] = self.penalties[[self.real_state_size - 1], :]
        else:
        # occurrence
            self.occurrence[[self.real_state_size - 1], :] = self.occurrence[[pos_a], :]
            self.occurrence[[pos_a], :] = self.occurrence[[pos_b], :]
            self.occurrence[[pos_b], :] = self.occurrence[[self.real_state_size - 1], :]

            self.alpha_presence_vector[[self.real_state_size - 1], :] = self.alpha_presence_vector[[pos_a], :]
            self.alpha_presence_vector[[pos_a], :] = self.alpha_presence_vector[[pos_b], :]
            self.alpha_presence_vector[[pos_b], :] = self.alpha_presence_vector[[self.real_state_size - 1], :]

            self.frequency_penalty_tensor[[self.real_state_size - 1], :] = self.frequency_penalty_tensor[[pos_a], :]
            self.frequency_penalty_tensor[[pos_a], :] = self.frequency_penalty_tensor[[pos_b], :]
            self.frequency_penalty_tensor[[pos_b], :] = self.frequency_penalty_tensor[[self.real_state_size - 1], :]

            self.penalty_decay_tensor[[self.real_state_size - 1], :] = self.penalty_decay_tensor[[pos_a], :]
            self.penalty_decay_tensor[[pos_a], :] = self.penalty_decay_tensor[[pos_b], :]
            self.penalty_decay_tensor[[pos_b], :] = self.penalty_decay_tensor[[self.real_state_size - 1], :]

            # sample params
            self.temperature_tensor[[self.real_state_size - 1], :] = self.temperature_tensor[[pos_a], :]
            self.temperature_tensor[[pos_a], :] = self.temperature_tensor[[pos_b], :]
            self.temperature_tensor[[pos_b], :] = self.temperature_tensor[[self.real_state_size - 1], :]

            self.top_p_tensor[[self.real_state_size - 1], :] = self.top_p_tensor[[pos_a], :]
            self.top_p_tensor[[pos_a], :] = self.top_p_tensor[[pos_b], :]
            self.top_p_tensor[[pos_b], :] = self.top_p_tensor[[self.real_state_size - 1], :]

            self.top_k_tensor[[self.real_state_size - 1], :] = self.top_k_tensor[[pos_a], :]
            self.top_k_tensor[[pos_a], :] = self.top_k_tensor[[pos_b], :]
            self.top_k_tensor[[pos_b], :] = self.top_k_tensor[[self.real_state_size - 1], :]

            self.presence_penalty_tensor[[self.real_state_size - 1], :] = self.presence_penalty_tensor[[pos_a], :]
            self.presence_penalty_tensor[[pos_a], :] = self.presence_penalty_tensor[[pos_b], :]
            self.presence_penalty_tensor[[pos_b], :] = self.presence_penalty_tensor[[self.real_state_size - 1], :]

    def _organize_batch(self):
        """返回 ([start_pos, end_pos),)

        - 0: forward one
        - 1: forward one suspended
        - 2: seq prefill
        - 3: finish"""
        current_task_list = [None] * self.max_batch_size

        for slot_pos, task_data in sorted(self.state_slot.items()):
            current_task_list[slot_pos] = task_data["state_category"]

        swarps, offsets = min_swaps_to_target_fast(current_task_list, [i for i in sorted(StateCategory)])

        for pos_a, pos_b in swarps:
            self._switch_batch(pos_a, pos_b)
            self.state_slot[pos_a], self.state_slot[pos_b] = self.state_slot[pos_b], self.state_slot[pos_a]

        return offsets

    def _process_events(self) -> bool:
        """
        处理事件队列中的所有事件

        Returns:
            是否需要关闭 Worker
        """
        # 批量拉取所有可用事件
        while True:
            try:
                event = self.master_event_queue.get_nowait()

                if event.get("type") == "shutdown":
                    self.shutdown_flag = True
                    return True
                # 其他事件类型可以在这里处理

            except queue.Empty:
                break

        return False

    def _handle_forward_seq(self, task_data: TaskData, slot_pos):
        assert task_data["is_prefilling"] == True
        assert task_data["next_input_token"] != None, "next_input_token shall not be None."

        if task_data["task"].cache_prefill and len(task_data["task"].prefill_tokens) == (
            max(task_data["task"].cache_prefill_padding - 1, 0)
        ):
            # print(
            #     "cache_prefill fwd seq",
            #     task_data["prefilled_tokens"],
            #     self.tokenizer.decode(task_data["prefilled_tokens"], utf8_errors="ignore"),
            # )
            task_data["state_category"] = StateCategory.FORWARD_ONE

            if task_data["task"].cache_prefill:
                task_data["task"].output_queue.put_nowait(
                    (
                        "cache_prefill",
                        {
                            "state": [
                                self.batch_state[0][:, :, [slot_pos], :].to(device="cpu", non_blocking=True),
                                self.batch_state[1][:, [slot_pos], :, :].to(device="cpu", non_blocking=True),
                                self.batch_state[2][[slot_pos]].to(device="cpu", non_blocking=True),
                            ],
                            "prefilled_tokens": tuple(task_data["prefilled_tokens"]),
                        },
                    )
                )
                task_data["prefill_cached"] = True

        if len(task_data["task"].prefill_tokens) == 0:
            task_data["state_category"] = StateCategory.FORWARD_ONE
            task_data["is_prefilling"] = False

        elif len(task_data["task"].prefill_tokens) < self.min_forward_seq_len:
            task_data["state_category"] = StateCategory.FORWARD_ONE
        else:
            # task_data["state_category"] = StateCategory.FORWARD_SEQ
            pass

    def _handle_forward_one_prefill_phase(self, task_data: TaskData, slot_pos: int):
        """处理 Prefill 阶段"""
        task = task_data["task"]

        task_data["prefilled_tokens"].append(task_data["next_input_token"])
        task_data["next_input_token"] = task.prefill_tokens.pop(0)
        if len(task.prefill_tokens) == 0:
            task_data["is_prefilling"] = False

        if (
            task_data["task"].cache_prefill
            and len(task_data["task"].prefill_tokens) == (max(task_data["task"].cache_prefill_padding - 1, 0))
            and not task_data["prefill_cached"]
        ):
            # print("cache_prefill fwd one", task_data["task"].prefill_tokens)
            task.output_queue.put_nowait(
                (
                    "cache_prefill",
                    {
                        "state": [
                            self.batch_state[0][:, :, [slot_pos], :].to(device="cpu", non_blocking=True),
                            self.batch_state[1][:, [slot_pos], :, :].to(device="cpu", non_blocking=True),
                            self.batch_state[2][[slot_pos]].to(device="cpu", non_blocking=True),
                        ],
                        "prefilled_tokens": tuple(task_data["prefilled_tokens"]),
                    },
                )
            )
            task_data["prefill_cached"] = True

    def _handle_forward_one_decode_phase(self, task_data: TaskData, slot_pos: int):
        """处理 Decode 阶段"""
        task = task_data["task"]
        new_token = task_data["new_token"]

        if new_token in task.stop_tokens:
            task.request_status = RequestStatus.FINISHED_STOPPED
            return
        else:

            new_text = self.tokenizer.decode([new_token], utf8_errors="ignore")  # TODO: 处理不完整的 utf8

            task.generated_tokens.append(new_token)
            task.decoded_texts.append(new_text)

            task.output_queue.put_nowait(("token_generated", (new_token, new_text)))

        if len(task.generated_tokens) >= task.max_tokens:
            task.request_status = RequestStatus.FINISHED_LENGTH_CAPPED
        else:
            if not self.enable_rapid_sampling:
                www = 0.0 if new_token in self.no_penalty_token_ids else 1.0
                self.occurrence[slot_pos, new_token] += www
                self.alpha_presence_vector[[slot_pos], [new_token]] = self.presence_penalty_tensor[[slot_pos], :]

            task_data["next_input_token"] = new_token

    def _is_task_aborted(self, task_data: TaskData):
        """检查任务是否打断"""
        task = task_data["task"]

        # 检查任务事件队列中是否有 abort 事件
        try:
            # while not task.task_event_queue.empty():
            event_type, payload = task.task_event_queue.get_nowait()
            if event_type == "abort":
                return True
        except queue.Empty:
            pass

        return False

    def _process_accomplished_tasks(self, accomplished_task_slot_pos: List[int]):
        """处理已完成的任务"""

        if not accomplished_task_slot_pos:
            return

        for slot in accomplished_task_slot_pos:
            self.state_slot[slot]["task"].output_queue.put_nowait(
                ("task_completed", self.state_slot[slot]["task"])
            )

            self.state_slot[slot] = {
                "task": None,
                "is_prefilling": None,
                "new_token": None,
                "next_input_token": None,
                "state_category": StateCategory.EMPTY,
                "prefilled_tokens": [],
            }

    def _fill_task_pool(self):
        """填充任务池直到达到 batch_size"""
        prefill_count = 0
        for slot_pos in range(self.max_batch_size):
            if prefill_count >= self.max_prefill_count:
                break

            if self.state_slot[slot_pos]["state_category"] != StateCategory.EMPTY:
                if self.state_slot[slot_pos]["state_category"] == StateCategory.FORWARD_SEQ:
                    prefill_count += 1
                continue
            try:
                prefill_count += 1
                task: Task = self.task_queue.get_nowait()

                # 处理任务状态
                if task.state is None:
                    # 初始化空状态 - 创建与当前 batch_size 兼容的零状态
                    new_state = self.model.generate_zero_state(1)
                else:
                    # 将状态移动到 GPU
                    new_state = [state.cuda() for state in task.state]

                device = torch.device("cuda")

                self.batch_state[0][:, :, [slot_pos], :] = new_state[0]
                self.batch_state[1][:, [slot_pos], :, :] = new_state[1]
                self.batch_state[2][[slot_pos]] = new_state[2]

                if self.enable_rapid_sampling:
                    # Rapid-Sampling penalties 初始化
                    self.penalties[[slot_pos], :] = torch.zeros(
                        (1, self.model_config.vocab_size),
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    self.occurrence[[slot_pos], :] = torch.zeros(
                        (1, self.model_config.vocab_size),
                        dtype=torch.float32,
                        device=self.batch_state[0].device,
                    )

                    self.alpha_presence_vector[[slot_pos], :] = torch.zeros(
                        (1, self.model_config.vocab_size),
                        dtype=torch.float32,
                        device=self.batch_state[0].device,
                    )

                    self.temperature_tensor[[slot_pos], :] = torch.tensor(
                        [[task.temperature if task.temperature > 0 else 1.0]],
                        dtype=torch.float16,
                        device=device,
                    )
                    self.top_p_tensor[[slot_pos], :] = torch.tensor(
                        [[task.top_p]],
                        dtype=torch.float16,
                        device=device,
                    )
                    self.top_k_tensor[[slot_pos], :] = torch.tensor(
                        [[task.top_k]],
                        dtype=torch.int32,
                        device=device,
                    )
                    self.frequency_penalty_tensor[[slot_pos], :] = torch.tensor(
                        [[task.frequency_penalty]],
                        dtype=torch.float16,
                        device=device,
                    )
                    self.penalty_decay_tensor[[slot_pos], :] = torch.tensor(
                        [[task.penalty_decay]],
                        dtype=torch.float16,
                        device=device,
                    )
                    self.presence_penalty_tensor[[slot_pos], :] = torch.tensor(
                        [[task.presence_penalty]],
                        dtype=torch.float32,
                        device=device,
                    )

                # 添加到 task_pool

                next_input_token = task.prefill_tokens.pop(0)

                if len(task.prefill_tokens) == 0:
                    state_category = StateCategory.FORWARD_ONE
                    is_prefilling = False
                elif len(task.prefill_tokens) - max((task.cache_prefill_padding - 1), 0) < self.min_forward_seq_len:
                    state_category = StateCategory.FORWARD_ONE
                    is_prefilling = True
                else:
                    state_category = StateCategory.FORWARD_SEQ
                    is_prefilling = True

                task_data: TaskData = {
                    "task": task,
                    "is_prefilling": is_prefilling,
                    "new_token": None,
                    "next_input_token": next_input_token,
                    "state_category": state_category,
                    "prefilled_tokens": [],
                    "prefill_cached": False,
                }
                self.state_slot[slot_pos] = task_data

            except queue.Empty:
                break

    def _run_forward_one(self, decode_offset: Tuple[int, int]):
        """运行模型前向推理，单 token ，适合 decode 和 prefill 模式"""

        # 构建批处理输入

        next_tokens = [None] * (decode_offset[1] - decode_offset[0])

        for slot_pos in range(*decode_offset):
            next_tokens[slot_pos - decode_offset[0]] = [self.state_slot[slot_pos]["next_input_token"]]

        decode_slice = slice(decode_offset[0], decode_offset[1])

        forward_state = [
            self.batch_state[0][:, :, decode_slice, :],
            self.batch_state[1][:, decode_slice, :, :],
            self.batch_state[2][decode_slice],
        ]

        # print("fo", next_tokens)

        # 模型前向传播
        # x1 = time.perf_counter()
        out = self.model.forward_seq_batch(next_tokens, forward_state)
        # x2 = time.perf_counter()
        # print(f"forward time: {(x2-x1):.4f}")

        # 处理禁止 token
        for slot_pos in range(*decode_offset):
            for forbidden_token in self.state_slot[slot_pos]["task"].forbidden_tokens:
                out[slot_pos - decode_offset[0]][forbidden_token] -= 1e10

        if self.enable_rapid_sampling:
            # 使用 Rapid-Sampling 进行采样
            # rapid-sampling 使用默认参数
            temperature = 1.0
            top_k = 0  # 0 或 -1 表示不限制
            top_p = 0.3
            presence_penalty = 0.5
            repetition_penalty = 0.5
            penalty_decay = 0.996

            # 注意：rapid_sampling_states 是一个字节数组，每个 batch 元素占用固定字节
            # 由于 _organize_batch 确保 decode 任务从索引 0 开始，decode_offset[0] 应为 0
            # 这里使用完整的 states，内核会根据 logits 的 batch size 自动处理
            batch_size = decode_offset[1] - decode_offset[0]
            
            new_tokens = self.rapid_sampling_module.batch_sampling_repetition_temperature_topk_topp(
                out.float().contiguous(),
                self.penalties[decode_slice, :].contiguous(),
                self.rapid_sampling_states,  # 使用完整的 states
                presence_penalty,
                repetition_penalty,
                penalty_decay,
                temperature,
                top_k,
                top_p,
            )
            
        else:
            # 原始采样逻辑
            self.occurrence[decode_slice, :] *= self.penalty_decay_tensor[decode_slice, :]
            out -= (
                self.alpha_presence_vector[decode_slice, :]
                + self.occurrence[decode_slice, :] * self.frequency_penalty_tensor[decode_slice, :]
            )

            # 采样
            new_tokens = sample_logits_real_batch(
                out,
                self.temperature_tensor[decode_slice, :],
                self.top_p_tensor[decode_slice, :],
                self.top_k_tensor[decode_slice, :],
            )

        for slot_pos in range(*decode_offset):
            new_token = new_tokens[slot_pos - decode_offset[0]].item()
            self.state_slot[slot_pos]["new_token"] = new_token

        del out

    def _run_forward_seq(self, seq_perfill_offset: Tuple[int, int]):
        """运行模型前向推理，token 序列，适合 prefill 模式"""
        token_seq_len_list = [
            (
                len(self.state_slot[i]["task"].prefill_tokens)
                - max((self.state_slot[i]["task"].cache_prefill_padding - 1), 0)
            )
            for i in range(*seq_perfill_offset)
        ]
        token_seq_len = min(self.max_forward_seq_len_per_forward, *token_seq_len_list)
        assert token_seq_len > 0

        next_tokens: List[List[int]] = [None] * (seq_perfill_offset[1] - seq_perfill_offset[0])
        for slot_pos in range(*seq_perfill_offset):
            slot_next_tokens = [self.state_slot[slot_pos]["next_input_token"]] + self.state_slot[slot_pos][
                "task"
            ].prefill_tokens[: token_seq_len - 1]
            self.state_slot[slot_pos]["prefilled_tokens"].extend(slot_next_tokens)

            next_tokens[slot_pos - seq_perfill_offset[0]] = slot_next_tokens
            self.state_slot[slot_pos]["task"].prefill_tokens = self.state_slot[slot_pos]["task"].prefill_tokens[
                token_seq_len - 1 :
            ]
            self.state_slot[slot_pos]["next_input_token"] = self.state_slot[slot_pos]["task"].prefill_tokens.pop(0)

        seq_forward_state = [
            self.batch_state[0][:, :, seq_perfill_offset[0] : seq_perfill_offset[1], :],
            self.batch_state[1][:, seq_perfill_offset[0] : seq_perfill_offset[1], :, :],
            self.batch_state[2][seq_perfill_offset[0] : seq_perfill_offset[1]],
        ]
        # print("fs", next_tokens)
        out = self.model.forward_batch(next_tokens, seq_forward_state)
        del out

    def start(self):
        """
        启动 Worker 的主循环

        这是 Worker 的核心方法，实现了 continuous batching 的完整流程。
        """
        # 初始化模型和状态
        if self.model is None:
            self._init_worker()

        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()

        # 主循环
        while True:
            loop_start_time = time.perf_counter()

            should_shutdown = self._process_events()

            if should_shutdown:
                break

            accomplished_task_slot_pos: list[int] = []

            for key, task_data in sorted(self.state_slot.items()):

                assert (
                    task_data["state_category"] != StateCategory.FINISHED
                ), f"Invalid state category: {task_data['state_category'] }"

                if task_data["state_category"] == StateCategory.EMPTY:
                    continue

                if self._is_task_aborted(task_data):
                    task_data["task"].request_status = RequestStatus.FINISHED_ABORTED
                    task_data["state_category"] = StateCategory.FINISHED

                elif task_data["state_category"] == StateCategory.FORWARD_SEQ:
                    self._handle_forward_seq(task_data, key)

                elif task_data["state_category"] == StateCategory.FORWARD_ONE:
                    if task_data["is_prefilling"]:
                        self._handle_forward_one_prefill_phase(task_data, key)
                    else:
                        self._handle_forward_one_decode_phase(task_data, key)

                if RequestStatus.is_finished(task_data["task"].request_status):
                    accomplished_task_slot_pos.append(key)

            self._process_accomplished_tasks(accomplished_task_slot_pos)

            self._fill_task_pool()

            decode_offset, decode_suspended_offset, seq_perfill_offset, accomplished_offset, empty_offset = (
                self._organize_batch()
            )

            if decode_offset[1] - decode_offset[0] == 0 and seq_perfill_offset[1] - seq_perfill_offset[0] == 0:
                time.sleep(0.05)
                continue

            if decode_offset[1] - decode_offset[0] > 0:
                self._run_forward_one(decode_offset)
                self.seq_forward_count_down -= 1
            else:
                self.seq_forward_count_down = 0

            if self.seq_forward_count_down < 1 and seq_perfill_offset[1] - seq_perfill_offset[0] > 0:
                self._run_forward_seq(seq_perfill_offset)
                self.seq_forward_count_down = max(1, self.decode_prefill_ratio)

            self.loop_time_recorder.append(time.perf_counter() - loop_start_time)

            if self.worker_event_queue:
                decode_count = sum([1 for i in range(*decode_offset) if self.state_slot[i]["is_prefilling"] == False])
                info = (
                    self.worker_id,
                    "worker_performance",
                    {
                        "avg_loop_time": sum(self.loop_time_recorder) / len(self.loop_time_recorder),
                        "state_size": self.real_state_size,
                        "state_offset_details": {
                            "decode_offset": decode_offset,
                            "decode_suspended_offset": decode_suspended_offset,
                            "seq_perfill_offset": seq_perfill_offset,
                            "accomplished_offset": accomplished_offset,
                        },
                        "task_details": {
                            "decode_count": decode_count,
                            "one_prefill_count": decode_offset[1] - decode_offset[0] - decode_count,
                            "seq_perfill_count": seq_perfill_offset[1] - seq_perfill_offset[0],
                        },
                        "max_allocated_memory_GB": torch.cuda.max_memory_allocated() / 1024**3,
                    },
                )
                self.worker_event_queue.put_nowait(info)

        profiler.stop()
        profiler.write_html(f"{self.worker_id}_bsz_80.html")

        self._cleanup()

    def _cleanup(self):
        """清理资源"""
        del self.state_slot
        del self.batch_state
        del self.occurrence
        del self.alpha_presence_vector
        del self.temperature_tensor
        del self.top_p_tensor
        del self.top_k_tensor
        del self.frequency_penalty_tensor
        del self.penalty_decay_tensor
        del self.presence_penalty_tensor
        del self.model

        # Rapid-Sampling 清理
        if self.enable_rapid_sampling:
            del self.penalties
            del self.rapid_sampling_states
            self.rapid_sampling_module = None
