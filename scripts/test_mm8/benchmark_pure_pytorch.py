import torch
import sys

# ========== 配置 ==========
B, N, M = 16, 4096, 16384  # batch, hidden, output
WARMUP, REPEATS = 50, 200
DEVICE = "cuda"
DTYPE = torch.float16

# ========== 量化函数（与 RWKV 官方一致）==========
def quantize_weight(w_fp16):
    w = w_fp16.float()
    mx = torch.amin(w, dim=0)
    w = w - mx
    my = torch.amin(w, dim=1, keepdim=True)  # [N, 1]
    w = w - my
    rx = torch.amax(w, dim=0)                # [M]
    w = w / rx
    ry = torch.amax(w, dim=1, keepdim=True)  # [N, 1]
    w = w / ry
    w = torch.clip(torch.floor(w * 256), 0, 255).to(torch.uint8)
    mx = mx.to(DTYPE).contiguous()
    rx = (rx / 16).to(DTYPE).contiguous()
    my = my.to(DTYPE).contiguous()
    ry = (ry / 16).to(DTYPE).contiguous()
    return w, mx, rx, my, ry

# ========== 原始 w8a16 算法（RWKV pip 官方实现）==========
def original_mm8(x, w, mx, rx, my, ry):
    # ry: [N,1], rx: [M], my: [N,1], mx: [M]
    # 广播路径: (w[N,M] + 0.5) * ry[N,1] * rx[M] + my[N,1] + mx[M]
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# ========== 优化 w8a16 算法（Albatross 分解法）==========
def optimized_mm8(x, w, mx, rx, my, ry):
    """
    Albatross分解法（经小规模数值验证严格等价）
    关键：ry/my 从 [N,1] → squeeze → [N] 向量，与 x[B,N] 特征维精准对齐
    """
    ry_vec = ry.squeeze(1)  # [N,1] → [N]  ✓ 与x第二维对齐
    my_vec = my.squeeze(1)  # [N,1] → [N]
    
    xs = x * ry_vec                    # [B, N] * [N] → [B, N]（ry沿batch广播）
    core = xs @ w.to(dtype=x.dtype)   # [B, M] ← 核心GEMM（占90%+耗时）
    xs_sum = xs.sum(dim=1, keepdim=True)      # [B, 1]
    xmy_sum = (x * my_vec).sum(dim=1, keepdim=True)  # [B, 1]
    x_sum = x.sum(dim=1, keepdim=True)        # [B, 1]
    
    # 重组：所有项自动广播至 [B, M]
    # (core + 0.5*xs_sum) * rx: [B,M] * [M] → [B,M]
    # xmy_sum: [B,1] → 广播至 [B,M]（每列相同）
    # x_sum * mx: [B,1] * [M] → [B,M]
    return ((core + 0.5 * xs_sum) * rx + xmy_sum + x_sum * mx).to(dtype=x.dtype)

# ========== Benchmark 工具 ==========
def benchmark(fn, label, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms → μs
    
    times.sort()
    trim = max(1, repeats // 10)
    trimmed = times[trim:-trim]
    avg = sum(trimmed) / len(trimmed)
    print(f"{label:<45} {avg:>10.1f} μs  (min={min(trimmed):.1f})")
    return avg

# ========== 精度验证 ==========
def check_precision(y_orig, y_opt, label):
    diff = (y_orig - y_opt).float()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_l2 = diff.norm() / y_orig.float().norm()
    cos_sim = torch.nn.functional.cosine_similarity(
        y_orig.float().view(1, -1), y_opt.float().view(1, -1)
    ).item()
    print(f"\n{label} 精度差异:")
    print(f"  Max Abs Error : {max_abs:.2e}")
    print(f"  Mean Abs Error: {mean_abs:.2e}")
    print(f"  Relative L2   : {rel_l2:.2e}")
    print(f"  Cosine Sim    : {cos_sim:.8f}")
    assert torch.allclose(y_orig, y_opt, rtol=1e-3, atol=1e-4), "精度差异超出容忍范围！"
    return max_abs

# ========== 主流程 ==========
def main():
    torch.manual_seed(42)
    print(f"测试配置: x[{B},{N}] @ w[{N},{M}]  (dtype={DTYPE}, device={DEVICE})")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"PyTorch 版本: {torch.__version__}\n")
    
    # 生成数据
    x = torch.randn(B, N, dtype=DTYPE, device=DEVICE)
    w_fp16 = torch.randn(N, M, dtype=DTYPE, device=DEVICE)
    w_q, mx, rx, my, ry = quantize_weight(w_fp16)
    
    # 验证形状 (关键!)
    print(f"张量形状验证:")
    print(f"  x:  {list(x.shape)}")
    print(f"  w:  {list(w_q.shape)} (uint8)")
    print(f"  mx: {list(mx.shape)}  rx: {list(rx.shape)}")
    print(f"  my: {list(my.shape)}  ry: {list(ry.shape)}  <-- 注意: ry/my 为 [N,1]\n")
    
    # FP16 基准
    print("FP16 基准耗时:")
    benchmark(lambda: x @ w_fp16, "FP16 baseline (@)")
    print()
    
    # ========== 1. 原始算法（无优化）==========
    y_orig = original_mm8(x, w_q, mx, rx, my, ry)
    print("原始 w8a16 算法 (无优化):")
    t_orig = benchmark(lambda: original_mm8(x, w_q, mx, rx, my, ry), "original_mm8 (eager)")
    
    # ========== 2. 优化算法（无优化）==========
    y_opt = optimized_mm8(x, w_q, mx, rx, my, ry)
    print("\n优化 w8a16 算法 (无优化):")
    t_opt = benchmark(lambda: optimized_mm8(x, w_q, mx, rx, my, ry), "optimized_mm8 (eager)")
    
    # ========== 3. JIT 编译优化 ==========
    print("\nJIT 编译优化:")
    try:
        jit_orig = torch.jit.script(original_mm8)
        _ = jit_orig(x, w_q, mx, rx, my, ry)
        t_jit_orig = benchmark(lambda: jit_orig(x, w_q, mx, rx, my, ry), "original_mm8 (JIT)")
    except Exception as e:
        print(f"  original_mm8 (JIT) 失败: {type(e).__name__}: {str(e)[:60]}")
        t_jit_orig = None
    
    try:
        jit_opt = torch.jit.script(optimized_mm8)
        _ = jit_opt(x, w_q, mx, rx, my, ry)
        t_jit_opt = benchmark(lambda: jit_opt(x, w_q, mx, rx, my, ry), "optimized_mm8 (JIT)")
    except Exception as e:
        print(f"  optimized_mm8 (JIT) 失败: {type(e).__name__}: {str(e)[:60]}")
        t_jit_opt = None
    
    # ========== 4. torch.compile 优化（Python 3.14+ 不支持，自动跳过）==========
    print("\ntorch.compile 优化:")
    t_compile_orig = t_compile_opt = None
    if sys.version_info < (3, 14):
        try:
            compiled_orig = torch.compile(original_mm8, mode="reduce-overhead")
            _ = compiled_orig(x, w_q, mx, rx, my, ry)
            t_compile_orig = benchmark(lambda: compiled_orig(x, w_q, mx, rx, my, ry), "original_mm8 (compile)")
        except Exception as e:
            print(f"  original_mm8 (compile) 失败: {type(e).__name__}: {str(e)[:60]}")
        
        try:
            compiled_opt = torch.compile(optimized_mm8, mode="reduce-overhead")
            _ = compiled_opt(x, w_q, mx, rx, my, ry)
            t_compile_opt = benchmark(lambda: compiled_opt(x, w_q, mx, rx, my, ry), "optimized_mm8 (compile)")
        except Exception as e:
            print(f"  optimized_mm8 (compile) 失败: {type(e).__name__}: {str(e)[:60]}")
    else:
        print("  跳过 torch.compile (Python 3.14+ 不支持)")
    
    # ========== 精度验证 ==========
    print("\n" + "="*70)
    print("精度验证: 优化算法 vs 原始算法 (eager mode)")
    print("="*70)
    max_err = check_precision(y_orig, y_opt, "优化 vs 原始")
    
    # ========== 速度总结 ==========
    print("\n" + "="*70)
    print("速度对比总结 (vs 原始 eager)")
    print("="*70)
    results = [
        ("原始 (eager)", t_orig),
        ("优化 (eager)", t_opt),
        ("原始 (JIT)", t_jit_orig),
        ("优化 (JIT)", t_jit_opt),
    ]
    if sys.version_info < (3, 14):
        results.extend([
            ("原始 (compile)", t_compile_orig),
            ("优化 (compile)", t_compile_opt),
        ])
    
    print(f"{'实现方案':<35} {'延迟(μs)':>12} {'加速比':>12}")
    print("-"*70)
    for name, t in results:
        if t is not None:
            speedup = t_orig / t
            print(f"{name:<35} {t:>12.1f} {speedup:>11.2f}x")
    print("-"*70)
    
    # 关键结论
    print(f"\n关键结论:")
    print(f"  • 优化算法 (eager) 比原始快 {t_orig/t_opt:.2f}x")
    if t_jit_opt:
        jit_gain = t_opt / t_jit_opt
        print(f"  • JIT 为优化算法额外带来 {jit_gain:.2f}x 加速 (vs eager)")
    if t_compile_opt and sys.version_info < (3, 14):
        compile_gain = t_jit_opt / t_compile_opt if t_jit_opt else t_opt / t_compile_opt
        print(f"  • compile 为优化算法额外带来 {compile_gain:.2f}x 加速 (vs JIT/eager)")
    print(f"  • 精度差异: MaxAbs={max_err:.2e} (安全范围内)")

if __name__ == "__main__":
    main()