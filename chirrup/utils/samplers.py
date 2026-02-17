from typing import List, Tuple, Dict, Any, Union

import torch
from torch.nn import functional as F

from collections import defaultdict

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
# MyModule = nn.Module
# def __nop(ob): return ob
# MyFunction = __nop
# MyStatic = __nop


@MyStatic
def sample_logits_real_batch_legacy(
    logits: torch.Tensor,
    temperature: torch.Tensor,  # [bsz, 1]
    top_p: torch.Tensor,  # [bsz, 1]
    top_k: torch.Tensor,  # [bsz, 1]
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """原始采样实现，保留用于验证和 fallback"""
    bsz, vocab_size = logits.shape

    # ====== 1. 温度缩放 (完全并行) ======
    # 直接广播 [bsz, 1] 的 temperature 到 [bsz, vocab_size]
    logits = logits / temperature

    # ====== 2. 计算概率分布 ======
    probs = F.softmax(logits.float(), dim=-1)

    # ====== 3. Top-k 过滤 (完全向量化) ======
    # 创建有效样本掩码 (top_k > 0)
    active_top_k = top_k > 0  # [bsz, 1]

    if active_top_k.any():
        # 调整 top_k 值: [min_tokens_to_keep, vocab_size]
        clamped_top_k = torch.clamp(top_k, min=min_tokens_to_keep, max=vocab_size).long()  # [bsz, 1]

        # 为无效样本设置虚拟值 (vocab_size 表示不过滤)
        effective_top_k = torch.where(
            active_top_k, clamped_top_k, torch.full_like(clamped_top_k, vocab_size)
        )  # [bsz, 1]

        # 生成排序索引和概率 [bsz, vocab_size]
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # 生成排名矩阵 [0, 1, ..., vocab_size-1] -> [bsz, vocab_size]
        ranks = torch.arange(vocab_size, device=logits.device).expand(bsz, vocab_size)

        # 创建移除掩码: 排名 >= effective_top_k 的位置
        remove_mask = ranks >= effective_top_k  # 广播 [bsz, 1] -> [bsz, vocab_size]

        # 应用掩码: 将低概率位置置零
        sorted_probs[remove_mask] = 0.0

        # 重建原始顺序的概率分布
        filtered_probs = torch.zeros_like(probs).scatter_(dim=1, index=sorted_indices, src=sorted_probs)

        # 更新概率 (仅活跃样本被修改)
        probs = torch.where(active_top_k.expand_as(probs), filtered_probs, probs)

    # ====== 4. Top-p 过滤 (完全向量化) ======
    # 为 top_p >= 1.0 的样本创建虚拟阈值 (2.0 确保不触发过滤)
    effective_top_p = torch.where(top_p < 1.0, top_p, torch.full_like(top_p, 2.0))  # [bsz, 1]

    # 生成排序索引和概率 [bsz, vocab_size]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [bsz, vocab_size]

    # 创建移除掩码: 累积概率 > effective_top_p
    sorted_remove = cumulative_probs > effective_top_p  # 广播 [bsz, 1] -> [bsz, vocab_size]

    # 确保至少保留 min_tokens_to_keep 个 token
    min_keep_mask = torch.arange(vocab_size, device=logits.device).expand(bsz, vocab_size) < min_tokens_to_keep
    sorted_remove = sorted_remove & (~min_keep_mask)

    # 转换到原始索引空间
    remove_mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(dim=1, index=sorted_indices, src=sorted_remove)

    # 应用掩码
    probs.masked_fill_(remove_mask, 0.0)

    # ====== 5. 归一化 (处理零概率和) ======
    probs_sum = probs.sum(dim=-1, keepdim=True)
    # 安全除法: 防止零概率和 (理论上 min_tokens_to_keep 会避免此情况)
    probs = probs / torch.where(probs_sum > 0, probs_sum, torch.ones_like(probs_sum))

    # ====== 6. 采样 (完全并行) ======
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_tokens


@MyStatic
def sample_logits_real_batch(
    logits: torch.Tensor,
    temperature: torch.Tensor,  # [bsz, 1]
    top_p: torch.Tensor,  # [bsz, 1]
    top_k: torch.Tensor,  # [bsz, 1]
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    优化版采样实现：合并 top-k 和 top-p 的排序操作
    
    优化点：
    1. 只进行一次排序
    2. 在排序空间内同时完成 top-k 和 top-p 过滤
    3. 最后只做一次 scatter 恢复原始索引（如果需要）
    """
    bsz, vocab_size = logits.shape

    # ====== 1. 温度缩放 ======
    logits = logits / temperature

    # ====== 2. 计算概率分布 ======
    probs = F.softmax(logits.float(), dim=-1)

    # ====== 3. 一次排序 ======
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # ====== 4. 在排序空间内同时进行 top-k 和 top-p 过滤 ======
    
    # 4.1 处理 top-k
    # top_k <= 0 表示不限制，设为 vocab_size
    clamped_top_k = torch.clamp(top_k, min=min_tokens_to_keep, max=vocab_size).long()
    effective_top_k = torch.where(top_k > 0, clamped_top_k, torch.full_like(clamped_top_k, vocab_size))
    
    # 生成排名矩阵 [bsz, vocab_size]
    ranks = torch.arange(vocab_size, device=logits.device).expand(bsz, vocab_size)
    
    # top-k 移除掩码: 排名 >= effective_top_k
    topk_remove_mask = ranks >= effective_top_k
    
    # 4.2 处理 top-p
    # 为 top_p >= 1.0 的样本创建虚拟阈值
    effective_top_p = torch.where(top_p < 1.0, top_p, torch.full_like(top_p, 2.0))
    
    # 累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # top-p 移除掩码: 累积概率 > effective_top_p
    topp_remove_mask = cumulative_probs > effective_top_p
    
    # 确保至少保留 min_tokens_to_keep 个 token
    min_keep_mask = ranks < min_tokens_to_keep
    topp_remove_mask = topp_remove_mask & (~min_keep_mask)
    
    # ====== 5. 合并掩码并应用 ======
    # 同时满足 top-k 或 top-p 任一条件即移除
    combined_remove_mask = topk_remove_mask | topp_remove_mask
    
    # 应用掩码
    sorted_probs = sorted_probs.masked_fill(combined_remove_mask, 0.0)
    
    # ====== 6. 归一化 ======
    probs_sum = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = sorted_probs / torch.where(probs_sum > 0, probs_sum, torch.ones_like(probs_sum))
    
    # ====== 7. 在排序空间采样，然后映射回原始索引 ======
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # [bsz, 1]
    
    # 通过 gather 获取原始 token id
    next_tokens = torch.gather(sorted_indices, dim=1, index=sampled_sorted_idx).squeeze(-1)
    
    return next_tokens


@MyStatic
def sample_logits_rwkv_pip_compatible(
    logits: torch.Tensor,
    temperature: torch.Tensor,  # [bsz, 1]
    top_p: torch.Tensor,  # [bsz, 1]
    top_k: torch.Tensor,  # [bsz, 1]
) -> torch.Tensor:
    """
    与 rwkv pip 包 PIPELINE.sample_logits 等价的批量采样实现。

    算法复现自:
    https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/utils.py

    关键算法顺序（与 rwkv pip 一致）：
    1. probs = softmax(logits)   — 不对 logits 做温度缩放
    2. top-p 过滤: 基于 cutoff 值（cumsum >= top_p 处的概率值）
    3. top-k 过滤: 保留前 k 个 token
    4. temperature 调整: probs = probs ** (1/T)  — 在过滤之后施加
    5. 归一化 + 采样
    """
    bsz, vocab_size = logits.shape

    # ====== 1. 处理 temperature=0 的特殊情况 ======
    # rwkv pip: temperature=0 → temperature=1.0, top_p=0（贪婪模式）
    zero_temp_mask = (temperature == 0)  # [bsz, 1]
    temperature = torch.where(zero_temp_mask, torch.ones_like(temperature), temperature)
    top_p = torch.where(zero_temp_mask, torch.zeros_like(top_p), top_p)

    # ====== 2. 计算概率分布（不做温度缩放） ======
    probs = F.softmax(logits.float(), dim=-1)  # [bsz, vocab_size]

    # ====== 3. 降序排序 ======
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # ====== 4. Top-p 过滤（基于 cutoff 值，与 rwkv pip 一致） ======
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [bsz, vocab_size]

    # 找到每个样本中 cumsum >= top_p 的第一个位置
    # searchsorted 在 sorted 序列中找到 top_p 应插入的位置
    # 由于 cumulative_probs 是递增的，可直接使用
    cutoff_positions = torch.searchsorted(cumulative_probs, top_p)  # [bsz, 1]
    cutoff_positions = torch.clamp(cutoff_positions, max=vocab_size - 1)

    # 获取 cutoff 处的概率值
    cutoff_values = torch.gather(sorted_probs, dim=1, index=cutoff_positions)  # [bsz, 1]

    # 将所有 < cutoff 的概率置零（在原始概率空间操作）
    probs = torch.where(probs < cutoff_values, torch.zeros_like(probs), probs)

    # ====== 5. Top-k 过滤 ======
    top_k_long = top_k.long()
    active_top_k = top_k_long > 0  # [bsz, 1]

    if active_top_k.any():
        # 使用原始 sorted_indices 来确定哪些 token 在 top-k 之外
        # 创建一个掩码: 对于每个样本，标记不在 top-k 的 token
        effective_k = torch.where(
            active_top_k, top_k_long, torch.full_like(top_k_long, vocab_size)
        )  # [bsz, 1]

        ranks = torch.arange(vocab_size, device=logits.device).expand(bsz, vocab_size)
        # 在排序空间中，排名 >= k 的位置应该被置零
        outside_topk_sorted = ranks >= effective_k  # [bsz, vocab_size]

        # 将排序空间的掩码转换到原始索引空间
        outside_topk = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            dim=1, index=sorted_indices, src=outside_topk_sorted
        )

        probs.masked_fill_(outside_topk, 0.0)

    # ====== 6. Temperature 调整（过滤之后施加，与 rwkv pip 一致） ======
    need_temp = (temperature != 1.0)  # [bsz, 1]
    if need_temp.any():
        # probs = probs ** (1.0 / temperature)
        inv_temp = 1.0 / temperature  # [bsz, 1]
        # 只对需要调整的样本施加，避免不必要计算
        # 对零概率位置 pow 是安全的 (0 ** x = 0 for x > 0)
        adjusted_probs = probs ** inv_temp
        probs = torch.where(need_temp.expand_as(probs), adjusted_probs, probs)

    # ====== 7. 采样 ======
    # torch.multinomial 会自动归一化
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_tokens
