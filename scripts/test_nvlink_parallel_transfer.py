import torch
import time
import sys

def check_environment():
    """检查环境是否满足要求"""
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        sys.exit(1)
    
    if torch.cuda.device_count() < 2:
        print("❌ 需要至少 2 个 GPU")
        sys.exit(1)
    
    # 检查 NVLink 连接（简化版，实际可用 nvidia-smi topo -m 验证）
    print(f"✅ 检测到 {torch.cuda.device_count()} 个 GPU")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

def parallel_bidirectional_transfer(size_mb=1024):
    """
    通过独立 CUDA 流实现 GPU 0 ↔ GPU 1 的并行双向传输
    """
    # 创建数据（约 size_mb MB）
    size = size_mb * 1024 * 1024 // 4  # float32 每个元素 4 字节
    data0 = torch.randn(size, device='cuda:0', dtype=torch.float32)
    data1 = torch.randn(size, device='cuda:3', dtype=torch.float32)
    
    # 创建目标张量
    recv0 = torch.empty_like(data0)  # GPU 0 接收来自 GPU 1 的数据
    recv1 = torch.empty_like(data1)  # GPU 1 接收来自 GPU 0 的数据
    
    # 创建独立 CUDA 流
    stream0 = torch.cuda.Stream(device='cuda:0')  # 用于 GPU0→GPU1 传输
    stream1 = torch.cuda.Stream(device='cuda:3')  # 用于 GPU1→GPU0 传输
    
    # 创建 CUDA 事件用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"📊 传输数据量: {size_mb} MB (每个方向)")
    print("=" * 60)
    
    # ========== 串行传输（基线）==========
    print("\n[1] 串行传输（基线）:")
    torch.cuda.synchronize()
    t0 = time.time()
    
    # GPU0 → GPU1
    recv1.copy_(data0)  # 阻塞操作
    # GPU1 → GPU0
    recv0.copy_(data1)  # 阻塞操作
    
    torch.cuda.synchronize()
    serial_time = time.time() - t0
    print(f"   耗时: {serial_time*1000:.2f} ms")
    print(f"   等效带宽: {2 * size_mb / serial_time:.2f} MB/s")
    
    # 验证数据正确性
    assert torch.allclose(recv0.cpu(), data1.cpu(), atol=1e-6), "GPU0 接收数据错误"
    assert torch.allclose(recv1.cpu(), data0.cpu(), atol=1e-6), "GPU1 接收数据错误"
    
    # ========== 并行传输（使用 CUDA 流）==========
    print("\n[2] 并行双向传输（CUDA 流 + non_blocking）:")
    
    # 重置接收缓冲区
    recv0.zero_()
    recv1.zero_()
    
    torch.cuda.synchronize()
    start_event.record()
    
    # 在独立流中启动非阻塞传输
    with torch.cuda.stream(stream0):
        # GPU0 → GPU1 (stream0 控制)
        recv1.copy_(data0, non_blocking=True)
    
    with torch.cuda.stream(stream1):
        # GPU1 → GPU0 (stream1 控制)
        recv0.copy_(data1, non_blocking=True)
    
    # 等待两个流完成
    stream0.synchronize()
    stream1.synchronize()
    
    end_event.record()
    end_event.synchronize()
    parallel_time = start_event.elapsed_time(end_event) / 1000  # 秒
    
    print(f"   耗时: {parallel_time*1000:.2f} ms")
    print(f"   等效带宽: {2 * size_mb / parallel_time:.2f} MB/s")
    
    # 验证数据正确性
    assert torch.allclose(recv0.cpu(), data1.cpu(), atol=1e-6), "GPU0 接收数据错误"
    assert torch.allclose(recv1.cpu(), data0.cpu(), atol=1e-6), "GPU1 接收数据错误"
    
    # ========== 性能对比 ==========
    print("\n" + "=" * 60)
    print(f"⏱️  性能对比:")
    print(f"   串行耗时: {serial_time*1000:.2f} ms")
    print(f"   并行耗时: {parallel_time*1000:.2f} ms")
    speedup = serial_time / parallel_time
    print(f"   ⚡ 加速比: {speedup:.2f}x")
    
    if speedup > 1.5:
        print("   ✅ 成功实现并行双向传输（NVLink 全双工特性生效）")
    else:
        print("   ⚠️  加速不明显（可能受 PCIe 限制或数据量太小）")
    
    print("=" * 60)

if __name__ == "__main__":
    check_environment()
    parallel_bidirectional_transfer(size_mb=34*6)  # 传输 2GB/方向