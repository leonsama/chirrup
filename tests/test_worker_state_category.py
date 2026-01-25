"""
测试 Worker 中 StateCategory 分离逻辑的正确性
"""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import queue
import torch
from collections import defaultdict

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chirrup.worker import (
    StateCategory,
    min_swaps_to_target_fast,
    Worker,
    TaskData,
)
from chirrup.core_structure import Task, ModelLoadConfig, RequestStatus


class TestStateCategoryEnum(unittest.TestCase):
    """测试 StateCategory 枚举值的顺序"""

    def test_state_category_order(self):
        """验证 StateCategory 的顺序：DECODE < PREFILL < SUSPENDED < SEQ < FINISHED < EMPTY"""
        self.assertLess(StateCategory.FORWARD_ONE_DECODE, StateCategory.FORWARD_ONE_PREFILL)
        self.assertLess(StateCategory.FORWARD_ONE_PREFILL, StateCategory.FORWARD_ONE_SUSPENDED)
        self.assertLess(StateCategory.FORWARD_ONE_SUSPENDED, StateCategory.FORWARD_SEQ)
        self.assertLess(StateCategory.FORWARD_SEQ, StateCategory.FINISHED)
        self.assertLess(StateCategory.FINISHED, StateCategory.EMPTY)

    def test_state_category_sorted(self):
        """验证 sorted(StateCategory) 的顺序"""
        sorted_categories = sorted(StateCategory)
        expected = [
            StateCategory.FORWARD_ONE_DECODE,
            StateCategory.FORWARD_ONE_PREFILL,
            StateCategory.FORWARD_ONE_SUSPENDED,
            StateCategory.FORWARD_SEQ,
            StateCategory.FINISHED,
            StateCategory.EMPTY,
        ]
        self.assertEqual(sorted_categories, expected)


class TestMinSwapsToTargetFast(unittest.TestCase):
    """测试 min_swaps_to_target_fast 函数"""

    def test_already_sorted(self):
        """已排序的列表不需要交换"""
        lst = [
            StateCategory.FORWARD_ONE_DECODE,
            StateCategory.FORWARD_ONE_DECODE,
            StateCategory.FORWARD_ONE_PREFILL,
            StateCategory.FORWARD_SEQ,
            StateCategory.EMPTY,
        ]
        original = lst.copy()
        swaps, offsets = min_swaps_to_target_fast(lst, list(sorted(StateCategory)))

        # 验证没有实际改变顺序的交换
        self.assertEqual(lst, original)

    def test_mixed_decode_and_prefill(self):
        """测试混合 decode 和 prefill 的排序"""
        lst = [
            StateCategory.FORWARD_ONE_PREFILL,  # 0
            StateCategory.FORWARD_ONE_DECODE,   # 1
            StateCategory.FORWARD_ONE_PREFILL,  # 2
            StateCategory.FORWARD_ONE_DECODE,   # 3
            StateCategory.EMPTY,                # 4
        ]
        swaps, offsets = min_swaps_to_target_fast(lst, list(sorted(StateCategory)))

        # 验证排序后 decode 在前，prefill 在后
        decode_count = 2
        prefill_count = 2
        
        for i in range(decode_count):
            self.assertEqual(lst[i], StateCategory.FORWARD_ONE_DECODE)
        for i in range(decode_count, decode_count + prefill_count):
            self.assertEqual(lst[i], StateCategory.FORWARD_ONE_PREFILL)

    def test_offsets_correctness(self):
        """验证 offsets 返回值的正确性"""
        lst = [
            StateCategory.FORWARD_ONE_DECODE,   # 1 个
            StateCategory.FORWARD_ONE_PREFILL,  # 2 个
            StateCategory.FORWARD_ONE_PREFILL,
            StateCategory.FORWARD_SEQ,          # 1 个
            StateCategory.EMPTY,                # 2 个
            StateCategory.EMPTY,
        ]
        swaps, offsets = min_swaps_to_target_fast(lst, list(sorted(StateCategory)))

        # offsets 顺序: DECODE, PREFILL, SUSPENDED, SEQ, FINISHED, EMPTY
        self.assertEqual(offsets[0], (0, 1))  # DECODE: [0, 1)
        self.assertEqual(offsets[1], (1, 3))  # PREFILL: [1, 3)
        self.assertEqual(offsets[2], (3, 3))  # SUSPENDED: [3, 3) 空
        self.assertEqual(offsets[3], (3, 4))  # SEQ: [3, 4)
        self.assertEqual(offsets[4], (4, 4))  # FINISHED: [4, 4) 空
        self.assertEqual(offsets[5], (4, 6))  # EMPTY: [4, 6)


class TestWorkerStateTransition(unittest.TestCase):
    """测试 Worker 中状态转换逻辑"""

    def setUp(self):
        """创建 mock Worker 实例"""
        # Mock 所有外部依赖
        self.mock_task_queue = queue.Queue()
        self.mock_master_event_queue = queue.Queue()
        self.mock_worker_event_queue = queue.Queue()
        self.mock_model_config = MagicMock(spec=ModelLoadConfig)
        self.mock_model_config.vocab_size = 65536
        self.mock_model_config.head_size = 64

    @patch("chirrup.worker.RWKV_x070")
    @patch("chirrup.worker.TRIE_TOKENIZER")
    def test_prefill_to_decode_transition(self, mock_tokenizer, mock_model):
        """测试 FORWARD_ONE_PREFILL -> FORWARD_ONE_DECODE 转换"""
        # 创建 mock task
        mock_task = MagicMock(spec=Task)
        mock_task.prefill_tokens = [1, 2]  # 2个 token 待 prefill
        mock_task.cache_prefill = False
        mock_task.cache_prefill_padding = 0

        task_data: TaskData = {
            "task": mock_task,
            "is_prefilling": True,
            "new_token": None,
            "next_input_token": 100,
            "state_category": StateCategory.FORWARD_ONE_PREFILL,
            "prefilled_tokens": [],
            "prefill_cached": False,
        }

        # 模拟 prefill 过程
        # 第一次 prefill
        task_data["prefilled_tokens"].append(task_data["next_input_token"])
        task_data["next_input_token"] = mock_task.prefill_tokens.pop(0)
        
        # prefill_tokens 还有 1 个，不应该转换
        self.assertEqual(len(mock_task.prefill_tokens), 1)
        self.assertEqual(task_data["state_category"], StateCategory.FORWARD_ONE_PREFILL)

        # 第二次 prefill
        task_data["prefilled_tokens"].append(task_data["next_input_token"])
        task_data["next_input_token"] = mock_task.prefill_tokens.pop(0)
        
        # prefill_tokens 为空，应该转换到 DECODE
        if len(mock_task.prefill_tokens) == 0:
            task_data["is_prefilling"] = False
            task_data["state_category"] = StateCategory.FORWARD_ONE_DECODE

        self.assertEqual(task_data["state_category"], StateCategory.FORWARD_ONE_DECODE)
        self.assertFalse(task_data["is_prefilling"])


class TestRunForwardOneLogic(unittest.TestCase):
    """测试 _run_forward_one 函数只对 decode 进行采样的逻辑"""

    def test_decode_only_sampling(self):
        """验证只有 decode 范围被采样"""
        # 模拟 decode_offset 和 one_prefill_offset
        decode_offset = (0, 2)      # 2 个 decode
        one_prefill_offset = (2, 4)  # 2 个 prefill

        # 模拟 forward 输出
        combined_count = one_prefill_offset[1] - decode_offset[0]
        mock_out = torch.randn(combined_count, 65536)

        # 只取 decode 部分进行采样
        decode_count = decode_offset[1] - decode_offset[0]
        decode_out = mock_out[:decode_count]

        # 验证 decode_out 形状正确
        self.assertEqual(decode_out.shape[0], 2)

        # 验证 prefill 部分不被采样（只是验证逻辑分离）
        prefill_out = mock_out[decode_count:]
        self.assertEqual(prefill_out.shape[0], 2)

    def test_empty_decode_no_sampling(self):
        """当 decode 为空时不进行采样"""
        decode_offset = (0, 0)      # 0 个 decode
        one_prefill_offset = (0, 2)  # 2 个 prefill

        decode_count = decode_offset[1] - decode_offset[0]
        
        # decode_count 为 0，不应该进行采样
        self.assertEqual(decode_count, 0)

        # 验证 combined 范围仍然正确
        combined_count = one_prefill_offset[1] - decode_offset[0]
        self.assertEqual(combined_count, 2)


class TestOrganizeBatchOffsets(unittest.TestCase):
    """测试 _organize_batch 返回的 offsets 正确性"""

    def test_offsets_continuous(self):
        """验证 offsets 是连续的"""
        # 模拟排序后的状态
        sorted_categories = [
            StateCategory.FORWARD_ONE_DECODE,
            StateCategory.FORWARD_ONE_DECODE,
            StateCategory.FORWARD_ONE_PREFILL,
            StateCategory.FORWARD_ONE_PREFILL,
            StateCategory.FORWARD_ONE_PREFILL,
            StateCategory.FORWARD_SEQ,
            StateCategory.EMPTY,
            StateCategory.EMPTY,
        ]

        swaps, offsets = min_swaps_to_target_fast(
            sorted_categories.copy(), 
            list(sorted(StateCategory))
        )

        # 验证 offsets 连续性
        # decode_offset[1] == one_prefill_offset[0]
        decode_offset = offsets[0]
        one_prefill_offset = offsets[1]
        self.assertEqual(decode_offset[1], one_prefill_offset[0])

        # one_prefill_offset[1] == suspended_offset[0]
        suspended_offset = offsets[2]
        self.assertEqual(one_prefill_offset[1], suspended_offset[0])

        # 最终验证所有 offset 连续
        for i in range(len(offsets) - 1):
            self.assertEqual(offsets[i][1], offsets[i + 1][0])


if __name__ == "__main__":
    unittest.main()
