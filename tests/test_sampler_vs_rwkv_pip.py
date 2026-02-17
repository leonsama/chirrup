"""
采样函数等价性测试：chirrup sampler vs rwkv pip package sampler

通过统计采样验证 chirrup 的批量采样实现
与 rwkv pip 包中 PIPELINE.sample_logits 产生等价的概率分布。

测试在 CPU 和 GPU（如可用）上同时运行。

rwkv pip 的 sample_logits 在 CPU 和 GPU 上有不同的代码路径：
- CPU 路径: numpy argsort + np.random.choice
- GPU 路径: torch.argsort + torch.multinomial

chirrup 的 sample_logits_rwkv_pip_compatible 使用统一的 torch 实现。
"""

import pytest
import torch
from collections import Counter

from rwkv.utils import PIPELINE, PIPELINE_ARGS
from chirrup.utils.samplers import sample_logits_rwkv_pip_compatible


VOCAB_SIZE = 1000

# 构建设备列表：始终包含 CPU，GPU 可用时也包含
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


def _rwkv_sample(logits: torch.Tensor, **kwargs) -> int:
    """调用 rwkv pip 的 sample_logits（不依赖 self.model）"""
    return PIPELINE.sample_logits(None, logits, **kwargs)


def _chirrup_sample(logits_1d: torch.Tensor, temperature: float = 1.0,
                     top_p: float = 0.85, top_k: int = 0) -> int:
    """将 1D logits 适配为 chirrup 批量采样接口（bsz=1），返回 token id"""
    device = logits_1d.device
    temp_t = torch.full((1, 1), temperature, device=device)
    top_p_t = torch.full((1, 1), top_p, device=device)
    top_k_t = torch.full((1, 1), top_k, dtype=torch.int32, device=device)
    return sample_logits_rwkv_pip_compatible(logits_1d.unsqueeze(0), temp_t, top_p_t, top_k_t).item()


def _collect_samples(logits, num_samples, temperature=1.0, top_p=0.85, top_k=0):
    """收集 rwkv 和 chirrup 各 num_samples 次采样结果"""
    rwkv_counter = Counter()
    chirrup_counter = Counter()

    for _ in range(num_samples):
        rwkv_counter[_rwkv_sample(logits.clone(), temperature=temperature, top_p=top_p, top_k=top_k)] += 1
        chirrup_counter[_chirrup_sample(logits.clone(), temperature=temperature, top_p=top_p, top_k=top_k)] += 1

    return rwkv_counter, chirrup_counter


def _max_freq_diff(counter_a, counter_b, num_samples):
    """计算两个频率分布的最大差异"""
    all_tokens = set(counter_a.keys()) | set(counter_b.keys())
    return max(
        abs(counter_a.get(t, 0) / num_samples - counter_b.get(t, 0) / num_samples)
        for t in all_tokens
    )


# ==================== 确定性采样对比 ====================


@pytest.mark.parametrize("device", _DEVICES)
def test_greedy_equivalence(device):
    """top_k=1 时应该返回相同的 token（确定性贪婪采样）"""
    torch.manual_seed(42)
    logits = torch.randn(VOCAB_SIZE, device=device) * 3.0

    rwkv_tokens = [
        _rwkv_sample(logits.clone(), temperature=1.0, top_p=1.0, top_k=1)
        for _ in range(20)
    ]
    chirrup_tokens = [
        _chirrup_sample(logits.clone(), temperature=1.0, top_p=1.0, top_k=1)
        for _ in range(20)
    ]

    for i in range(20):
        assert rwkv_tokens[i] == chirrup_tokens[i], (
            f"[{device}] 贪婪采样第 {i} 次结果不一致: rwkv={rwkv_tokens[i]}, chirrup={chirrup_tokens[i]}"
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_greedy_equals_argmax(device):
    """top_k=1 贪婪采样结果应等于 argmax"""
    torch.manual_seed(42)
    logits = torch.randn(VOCAB_SIZE, device=device) * 3.0
    expected = logits.argmax().item()

    rwkv_token = _rwkv_sample(logits.clone(), temperature=1.0, top_p=1.0, top_k=1)
    chirrup_token = _chirrup_sample(logits.clone(), temperature=1.0, top_p=1.0, top_k=1)

    assert rwkv_token == expected
    assert chirrup_token == expected


# ==================== 统计采样对比 ====================


@pytest.mark.parametrize("device", _DEVICES)
def test_statistical_distribution_t1(device):
    """temperature=1.0 时通过统计采样验证两个实现产生相同分布"""
    num_samples = 10000
    torch.manual_seed(42)
    logits = torch.randn(VOCAB_SIZE, device=device) * 3.0

    rwkv_counter, chirrup_counter = _collect_samples(logits, num_samples, top_p=0.9, top_k=50)
    max_diff = _max_freq_diff(rwkv_counter, chirrup_counter, num_samples)

    assert max_diff < 0.03, f"[{device}] 统计分布差异过大: max_diff={max_diff:.4f}"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("top_p", [0.3, 0.5, 0.85])
def test_statistical_distribution_various_top_p(device, top_p):
    """不同 top-p 值下统计分布对比"""
    num_samples = 5000
    torch.manual_seed(123)
    logits = torch.randn(VOCAB_SIZE, device=device) * 3.0

    rwkv_counter, chirrup_counter = _collect_samples(logits, num_samples, top_p=top_p, top_k=0)
    max_diff = _max_freq_diff(rwkv_counter, chirrup_counter, num_samples)

    assert max_diff < 0.04, f"[{device}] top_p={top_p}: 统计分布差异过大: max_diff={max_diff:.4f}"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("top_k", [5, 10, 50, 100])
def test_statistical_distribution_various_top_k(device, top_k):
    """不同 top-k 值下统计分布对比"""
    num_samples = 5000
    torch.manual_seed(42)
    logits = torch.randn(VOCAB_SIZE, device=device) * 3.0

    rwkv_counter, chirrup_counter = _collect_samples(logits, num_samples, top_p=1.0, top_k=top_k)
    max_diff = _max_freq_diff(rwkv_counter, chirrup_counter, num_samples)

    assert max_diff < 0.04, f"[{device}] top_k={top_k}: 统计分布差异过大: max_diff={max_diff:.4f}"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("top_p,top_k", [
    (0.5, 10),
    (0.7, 50),
    (0.9, 100),
    (0.3, 5),
])
def test_statistical_distribution_combined(device, top_p, top_k):
    """同时使用 top-k 和 top-p 的统计分布对比"""
    num_samples = 5000
    torch.manual_seed(42)
    logits = torch.randn(VOCAB_SIZE, device=device) * 3.0

    rwkv_counter, chirrup_counter = _collect_samples(logits, num_samples, top_p=top_p, top_k=top_k)
    max_diff = _max_freq_diff(rwkv_counter, chirrup_counter, num_samples)

    assert max_diff < 0.04, (
        f"[{device}] top_p={top_p}, top_k={top_k}: 统计分布差异过大: max_diff={max_diff:.4f}"
    )


# ==================== 边界条件 ====================


@pytest.mark.parametrize("device", _DEVICES)
def test_no_filter_sampling(device):
    """top_p=1.0, top_k=0 时无过滤，采样结果集合应一致"""
    num_samples = 3000
    torch.manual_seed(42)
    logits = torch.randn(VOCAB_SIZE, device=device) * 3.0

    rwkv_counter, chirrup_counter = _collect_samples(logits, num_samples, top_p=1.0, top_k=0)
    max_diff = _max_freq_diff(rwkv_counter, chirrup_counter, num_samples)

    assert max_diff < 0.03, f"[{device}] 无过滤时统计分布差异过大: max_diff={max_diff:.4f}"


@pytest.mark.parametrize("device", _DEVICES)
def test_single_dominant_logit(device):
    """极端 logits 分布（一个远大于其他）时两者应都选择该 token"""
    logits = torch.zeros(VOCAB_SIZE, device=device)
    logits[42] = 100.0

    for _ in range(20):
        rwkv_token = _rwkv_sample(logits.clone(), temperature=1.0, top_p=0.9, top_k=50)
        chirrup_token = _chirrup_sample(logits.clone(), temperature=1.0, top_p=0.9, top_k=50)
        assert rwkv_token == 42
        assert chirrup_token == 42


@pytest.mark.parametrize("device", _DEVICES)
def test_uniform_logits_sampling(device):
    """均匀 logits 分布下两者频率应接近均匀"""
    num_samples = 5000
    vocab_size = 100  # 用较小词表方便统计
    logits = torch.ones(vocab_size, device=device)

    rwkv_counter = Counter()
    chirrup_counter = Counter()
    temp_t = torch.full((1, 1), 1.0, device=device)
    top_p_t = torch.full((1, 1), 1.0, device=device)
    top_k_t = torch.full((1, 1), 0, dtype=torch.int32, device=device)

    for _ in range(num_samples):
        rwkv_counter[_rwkv_sample(logits.clone(), temperature=1.0, top_p=1.0, top_k=0)] += 1
        chirrup_counter[
            sample_logits_rwkv_pip_compatible(logits.clone().unsqueeze(0), temp_t, top_p_t, top_k_t).item()
        ] += 1

    expected_freq = 1.0 / vocab_size
    for token in range(vocab_size):
        rwkv_freq = rwkv_counter.get(token, 0) / num_samples
        chirrup_freq = chirrup_counter.get(token, 0) / num_samples
        assert abs(rwkv_freq - expected_freq) < 0.03, f"[{device}] rwkv token {token} 频率偏离均匀"
        assert abs(chirrup_freq - expected_freq) < 0.03, f"[{device}] chirrup token {token} 频率偏离均匀"


# ==================== temperature ≠ 1.0 ====================


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("temperature", [0.5, 0.8, 1.5, 2.0])
def test_temperature_nonzero_token_set_comparable(device, temperature):
    """
    temperature ≠ 1.0 时，采样的 token 集合仍应有大量重叠。
    """
    num_samples = 3000
    torch.manual_seed(42)
    logits = torch.randn(VOCAB_SIZE, device=device) * 3.0

    rwkv_counter, chirrup_counter = _collect_samples(
        logits, num_samples, temperature=temperature, top_p=0.9, top_k=0
    )

    rwkv_tokens = set(rwkv_counter.keys())
    chirrup_tokens = set(chirrup_counter.keys())

    overlap = len(rwkv_tokens & chirrup_tokens)
    total = len(rwkv_tokens | chirrup_tokens)
    jaccard = overlap / max(total, 1)

    assert jaccard > 0.5, (
        f"[{device}] temperature={temperature}: token 集合重叠过低 "
        f"(jaccard={jaccard:.2f}, rwkv={len(rwkv_tokens)}, chirrup={len(chirrup_tokens)})"
    )


# ==================== rwkv pip 包导入验证 ====================


def test_import_rwkv_utils():
    """测试 rwkv 包可以导入"""
    assert hasattr(PIPELINE, 'sample_logits')


def test_pipeline_args_defaults():
    """测试 PIPELINE_ARGS 默认参数"""
    args = PIPELINE_ARGS()
    assert args.temperature == 1.0
    assert args.top_p == pytest.approx(0.85)
    assert args.top_k == 0


def test_sample_logits_callable_without_model():
    """sample_logits 不依赖 self.model，可以直接调用"""
    logits = torch.randn(100)
    token = PIPELINE.sample_logits(None, logits, temperature=1.0, top_p=0.9, top_k=0)
    assert isinstance(token, int)
    assert 0 <= token < 100
