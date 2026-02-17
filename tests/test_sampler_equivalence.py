"""
采样函数等价性测试

通过统计方法验证新旧采样实现（legacy vs optimized）产生相同的概率分布。
测试在 CPU 和 GPU（如可用）上同时运行。
"""

import pytest
import torch
from collections import Counter

from chirrup.utils.samplers import sample_logits_real_batch, sample_logits_real_batch_legacy


BSZ = 4
VOCAB_SIZE = 1000

# 构建设备列表：始终包含 CPU，GPU 可用时也包含
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


def _create_tensors(device, temperature=1.0, top_p=0.9, top_k=50):
    """创建测试用的参数张量"""
    temp_tensor = torch.full((BSZ, 1), temperature, dtype=torch.float16, device=device)
    top_p_tensor = torch.full((BSZ, 1), top_p, dtype=torch.float16, device=device)
    top_k_tensor = torch.full((BSZ, 1), top_k, dtype=torch.int32, device=device)
    return temp_tensor, top_p_tensor, top_k_tensor


# ==================== 基础功能测试 ====================


@pytest.mark.parametrize("device", _DEVICES)
def test_output_shape(device):
    """测试输出形状正确"""
    logits = torch.randn(BSZ, VOCAB_SIZE, device=device)
    temp, top_p, top_k = _create_tensors(device)

    legacy_tokens = sample_logits_real_batch_legacy(logits.clone(), temp, top_p, top_k)
    new_tokens = sample_logits_real_batch(logits.clone(), temp, top_p, top_k)

    assert legacy_tokens.shape == (BSZ,)
    assert new_tokens.shape == (BSZ,)


@pytest.mark.parametrize("device", _DEVICES)
def test_output_range(device):
    """测试输出范围正确"""
    logits = torch.randn(BSZ, VOCAB_SIZE, device=device)
    temp, top_p, top_k = _create_tensors(device)

    for _ in range(100):
        legacy_tokens = sample_logits_real_batch_legacy(logits.clone(), temp, top_p, top_k)
        new_tokens = sample_logits_real_batch(logits.clone(), temp, top_p, top_k)

        assert (legacy_tokens >= 0).all()
        assert (legacy_tokens < VOCAB_SIZE).all()
        assert (new_tokens >= 0).all()
        assert (new_tokens < VOCAB_SIZE).all()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("temperature,top_p,top_k", [
    (1.0, 1.0, 0),     # 无过滤
    (0.5, 1.0, 0),     # 低温度
    (1.0, 0.1, 0),     # 极低 top-p
    (1.0, 0.5, 10),    # 组合 top-k 和 top-p
    (2.0, 0.9, 50),    # 高温度
])
def test_sampling_configs(device, temperature, top_p, top_k):
    """测试各种参数配置下输出形状正确"""
    logits = torch.randn(BSZ, VOCAB_SIZE, device=device)
    temp, top_p_t, top_k_t = _create_tensors(device, temperature=temperature, top_p=top_p, top_k=top_k)

    for _ in range(100):
        legacy_tokens = sample_logits_real_batch_legacy(logits.clone(), temp, top_p_t, top_k_t)
        new_tokens = sample_logits_real_batch(logits.clone(), temp, top_p_t, top_k_t)

        assert legacy_tokens.shape == (BSZ,)
        assert new_tokens.shape == (BSZ,)


@pytest.mark.parametrize("device", _DEVICES)
def test_greedy_mode_top_k_1(device):
    """测试贪婪模式 (top_k=1) 应该是确定性的"""
    torch.manual_seed(123)
    logits = torch.randn(BSZ, VOCAB_SIZE, device=device)
    temp, top_p, top_k = _create_tensors(device, temperature=1.0, top_p=1.0, top_k=1)

    results_legacy = []
    results_new = []

    for _ in range(10):
        legacy_tokens = sample_logits_real_batch_legacy(logits.clone(), temp, top_p, top_k)
        new_tokens = sample_logits_real_batch(logits.clone(), temp, top_p, top_k)
        results_legacy.append(legacy_tokens.clone())
        results_new.append(new_tokens.clone())

    # 验证所有结果相同（确定性）
    for i in range(1, 10):
        assert torch.equal(results_legacy[0], results_legacy[i])
        assert torch.equal(results_new[0], results_new[i])

    # 验证新旧结果一致
    assert torch.equal(results_legacy[0], results_new[0])


@pytest.mark.parametrize("device", _DEVICES)
def test_distribution_equivalence(device):
    """通过多次采样验证新旧实现产生相同的概率分布"""
    num_samples = 5000
    tolerance = 0.05

    torch.manual_seed(42)
    logits = torch.randn(BSZ, VOCAB_SIZE, device=device) * 3.0
    temp, top_p, top_k = _create_tensors(device, temperature=1.0, top_p=0.9, top_k=50)

    legacy_counts = [Counter() for _ in range(BSZ)]
    new_counts = [Counter() for _ in range(BSZ)]

    for _ in range(num_samples):
        legacy_tokens = sample_logits_real_batch_legacy(logits.clone(), temp, top_p, top_k)
        new_tokens = sample_logits_real_batch(logits.clone(), temp, top_p, top_k)

        for b in range(BSZ):
            legacy_counts[b][legacy_tokens[b].item()] += 1
            new_counts[b][new_tokens[b].item()] += 1

    # 比较分布
    for b in range(BSZ):
        all_tokens = set(legacy_counts[b].keys()) | set(new_counts[b].keys())

        for token in all_tokens:
            legacy_prob = legacy_counts[b].get(token, 0) / num_samples
            new_prob = new_counts[b].get(token, 0) / num_samples
            diff = abs(legacy_prob - new_prob)

            assert diff <= tolerance, (
                f"[{device}] Batch {b}, Token {token}: legacy_prob={legacy_prob:.4f}, "
                f"new_prob={new_prob:.4f}, diff={diff:.4f}"
            )


# ==================== 性能测试 ====================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_performance_comparison():
    """性能对比测试"""
    import time

    device = "cuda"
    bsz = 80
    vocab_size = 65536
    num_iterations = 100

    logits = torch.randn(bsz, vocab_size, device=device)
    temp = torch.full((bsz, 1), 1.0, dtype=torch.float16, device=device)
    top_p = torch.full((bsz, 1), 0.9, dtype=torch.float16, device=device)
    top_k = torch.full((bsz, 1), 50, dtype=torch.int32, device=device)

    # 预热
    for _ in range(10):
        sample_logits_real_batch_legacy(logits.clone(), temp, top_p, top_k)
        sample_logits_real_batch(logits.clone(), temp, top_p, top_k)

    torch.cuda.synchronize()

    # Legacy 版本
    start = time.perf_counter()
    for _ in range(num_iterations):
        sample_logits_real_batch_legacy(logits.clone(), temp, top_p, top_k)
    torch.cuda.synchronize()
    legacy_time = time.perf_counter() - start

    # 新版本
    start = time.perf_counter()
    for _ in range(num_iterations):
        sample_logits_real_batch(logits.clone(), temp, top_p, top_k)
    torch.cuda.synchronize()
    new_time = time.perf_counter() - start

    print(f"\n性能对比 (bsz={bsz}, vocab={vocab_size}, iters={num_iterations}):")
    print(f"  Legacy: {legacy_time*1000:.2f} ms ({legacy_time/num_iterations*1000:.2f} ms/iter)")
    print(f"  New:    {new_time*1000:.2f} ms ({new_time/num_iterations*1000:.2f} ms/iter)")
    print(f"  加速比: {legacy_time/new_time:.2f}x")
