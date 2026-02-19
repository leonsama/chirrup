"""
mm8 (int8 quantized matmul) performance benchmark
===================================================
Compares:
  0. [Baseline]   PyTorch native fp16 @ operator
  1. [rwkv pip]   cuBLAS gemm_fp16_cublas  (from ChatRWKV)
  2. [rwkv pip]   PyTorch fallback  (torch_mm8_one / torch_mm8_seq)
  3. [rwkv pip]   CUDA kernel       (cuda_mm8_one / cuda_mm8_seq from operators.cu)
  4. [Albatross]  PyTorch mm8       (torch_mm8_seq / torch_mm8_one from rwkv7.py)

Weight quantisation follows the rwkv pip package convention (w8a16).
Baseline: fp16 1×4096 @ fp16 4096×16384
"""

import os
import time
import torch
import torch.nn.functional as F

current_path = os.path.dirname(os.path.abspath(__file__))

# =============================== Compile CUDA extensions ===============================
print("Compiling CUDA extensions ...")
from torch.utils.cpp_extension import load

rwkv_pip_cuda = load(
    name="rwkv_pip_mm8",
    sources=[
        os.path.join(current_path, "rwkv_pip_wrapper.cpp"),
        os.path.join(current_path, "rwkv_pip_operators.cu"),
        os.path.join(current_path, "gemm_fp16_cublas.cpp"),
    ],
    verbose=True,
    extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
    extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
    is_python_module=False,
)
print("CUDA extensions compiled.\n")

# ============================ Config ============================

N = 4096       # activation inner dim
M = 16384      # weight output dim
DEVICE = "cuda"
DTYPE = torch.float16

WARMUP = 50
REPEATS = 200

# ======================= Weight quantisation (same as rwkv pip package) =======================

def quantize_weight(w_fp16):
    """
    Quantise an fp16 weight matrix to uint8 + (mx, rx, my, ry) exactly as the
    rwkv pip package does in model.py RWKV.__init__().
    """
    w = w_fp16.float()
    # w shape: [N, M].  N=4096 < M=16384  →  else branch in model.py
    if w.shape[0] > w.shape[1]:
        my = torch.amin(w, dim=1).unsqueeze(1)
        w = w - my
        mx = torch.amin(w, dim=0)
        w = w - mx
        rx = torch.amax(w, dim=0)
        w = w / rx
        ry = torch.amax(w, dim=1).unsqueeze(1)
        w = w / ry
    else:
        mx = torch.amin(w, dim=0)
        w = w - mx
        my = torch.amin(w, dim=1).unsqueeze(1)
        w = w - my
        rx = torch.amax(w, dim=0)
        w = w / rx
        ry = torch.amax(w, dim=1).unsqueeze(1)
        w = w / ry

    w = torch.clip(torch.floor(w * 256), min=0, max=255).to(dtype=torch.uint8)
    mx = mx.to(dtype=DTYPE).contiguous()
    rx = (rx / 16).to(dtype=DTYPE).contiguous()
    my = my.to(dtype=DTYPE).contiguous()
    ry = (ry / 16).to(dtype=DTYPE).contiguous()
    return w, mx, rx, my, ry

# ======================= Implementation 0: PyTorch native fp16 @ (baseline) =======================

def pytorch_fp16_matmul_seq(x_2d, w_fp16):
    return x_2d @ w_fp16

def pytorch_fp16_matmul_one(x_1d, w_fp16):
    return x_1d @ w_fp16

# ======================= Implementation 1: rwkv pip cuBLAS gemm_fp16_cublas =======================
# Copied from https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/cuda/gemm_fp16_cublas.cpp

def cublas_gemm_fp16_seq(x_2d, w_fp16):
    """x_2d: [B, N], w_fp16: [N, M] -> c: [B, M]"""
    c = torch.empty((x_2d.shape[0], w_fp16.shape[-1]), dtype=x_2d.dtype, device=x_2d.device)
    torch.ops.rwkv_pip.gemm_fp16_cublas(x_2d, w_fp16, c)
    return c

def cublas_gemm_fp16_one(x_1d, w_fp16):
    """x_1d: [N], w_fp16: [N, M] -> c: [M]"""
    x_2d = x_1d.unsqueeze(0)
    c = torch.empty((1, w_fp16.shape[-1]), dtype=x_1d.dtype, device=x_1d.device)
    torch.ops.rwkv_pip.gemm_fp16_cublas(x_2d, w_fp16, c)
    return c.squeeze(0)

# ======================= Implementation 2: rwkv pip PyTorch fallback =======================
# Copied from https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/model.py

def rwkv_pip_torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

def rwkv_pip_torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# ======================= Implementation 3: rwkv pip CUDA kernel =======================
# Compiled from rwkv_pip_operators.cu / rwkv_pip_wrapper.cpp
# (copied from https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/cuda/operators.cu
#  and       https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/cuda/wrapper.cpp)

def cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry):
    assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
    assert x.dtype == torch.float32 or x.dtype == torch.float16
    assert w.dtype == torch.uint8
    assert x.shape == (B, N)
    assert w.shape == (N, M)
    assert rx.shape == mx.shape == (M,)
    assert ry.shape == my.shape == (N, 1)
    y = torch.empty((B, M), device=w.device, dtype=x.dtype)
    torch.ops.rwkv_pip.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
    return y

def cuda_mm8_one(N, M, x, w, mx, rx, my, ry):
    assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
    assert x.dtype == torch.float32 or x.dtype == torch.float16
    assert w.dtype == torch.uint8
    assert x.shape == (N,)
    assert w.shape == (N, M)
    assert rx.shape == mx.shape == (M,)
    assert ry.shape == my.shape == (N, 1)
    y = torch.zeros((M,), device=w.device, dtype=torch.float32)
    torch.ops.rwkv_pip.mm8_one(N, M, x, w, mx, rx, my, ry, y)
    return y.to(dtype=x.dtype)

# ======================= Implementation 4: Albatross PyTorch mm8 =======================
# Copied from Albatross/rwkv7.py (mm8_seq_op -> torch_mm8_seq / mm8_one_op -> torch_mm8_one)

def albatross_torch_mm8_seq(x, w, mx, rx, my, ry):
    # Decomposed dequantization: only 1 [N,M] temporary (w type cast) instead of 5-6
    # y[k] = rx[k] * (sum_j xs[j]*w[j,k] + 0.5*xs_sum) + xmy_sum + mx[k]*x_sum
    ry_flat = ry.view(1, -1)   # [1, N]
    my_flat = my.view(1, -1)   # [1, N]
    rx_flat = rx.view(1, -1)   # [1, M]
    mx_flat = mx.view(1, -1)   # [1, M]
    xs = x * ry_flat                          # [B, N]
    core = xs @ w.to(dtype=x.dtype)           # [B, M] cuBLAS GEMM
    xs_sum = xs.sum(dim=-1, keepdim=True)     # [B, 1]
    x_sum = x.sum(dim=-1, keepdim=True)       # [B, 1]
    xmy_sum = (x * my_flat).sum(dim=-1, keepdim=True)  # [B, 1]
    return (rx_flat * (core + 0.5 * xs_sum) + xmy_sum + mx_flat * x_sum).to(dtype=x.dtype)

def albatross_torch_mm8_one(x, w, mx, rx, my, ry):
    # Decomposed dequantization: only 1 [N,M] temporary (w type cast) instead of 5-6
    # y[k] = rx[k] * (sum_j xs[j]*w[j,k] + 0.5*xs_sum) + xmy_sum + mx[k]*x_sum
    ry_flat = ry.view(-1)    # [N]
    my_flat = my.view(-1)    # [N]
    rx_flat = rx.view(-1)    # [M]
    mx_flat = mx.view(-1)    # [M]
    xs = x * ry_flat                    # [N]
    core = xs @ w.to(dtype=x.dtype)     # [M] cuBLAS GEMV
    xs_sum = xs.sum()                   # scalar
    x_sum = x.sum()                     # scalar
    xmy_sum = (x * my_flat).sum()       # scalar
    return (rx_flat * (core + 0.5 * xs_sum) + xmy_sum + mx_flat * x_sum).to(dtype=x.dtype)

# =============================== Benchmark utility ===============================

def benchmark(fn, warmup=WARMUP, repeats=REPEATS, label=""):
    """CUDA-event-based latency measurement (microseconds)."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # ms → μs
    times.sort()
    # trim 10% outliers from each end
    trim = max(1, repeats // 10)
    trimmed = times[trim:-trim]
    avg = sum(trimmed) / len(trimmed)
    mn  = min(trimmed)
    mx  = max(trimmed)
    return avg, mn, mx

# =============================== Main ===============================

def main():
    print("=" * 80)
    print("mm8 benchmark:  x[1,4096] @ w[4096,16384]   (fp16, CUDA)")
    print("=" * 80)

    torch.manual_seed(42)

    # ---- Create test data ----
    x_2d = torch.randn(1, N, dtype=DTYPE, device=DEVICE)    # [1, N]
    x_1d = x_2d.squeeze(0)                                   # [N]
    w_fp16 = torch.randn(N, M, dtype=DTYPE, device=DEVICE)   # [N, M]

    # Quantise
    w_q, mx, rx, my, ry = quantize_weight(w_fp16)
    w_q  = w_q.to(device=DEVICE).contiguous()
    mx   = mx.to(device=DEVICE)
    rx   = rx.to(device=DEVICE)
    my   = my.to(device=DEVICE)
    ry   = ry.to(device=DEVICE)

    print(f"\nShapes:  x_2d={list(x_2d.shape)}  x_1d={list(x_1d.shape)}")
    print(f"         w_fp16={list(w_fp16.shape)}  w_q={list(w_q.shape)} (uint8)")
    print(f"         mx={list(mx.shape)}  rx={list(rx.shape)}  my={list(my.shape)}  ry={list(ry.shape)}")
    print(f"\nWarmup={WARMUP}  Repeats={REPEATS}")
    print()

    results = []

    # ====== 0. Baseline: PyTorch native fp16 @ (seq) ======
    avg, mn, mx_t = benchmark(lambda: pytorch_fp16_matmul_seq(x_2d, w_fp16))
    results.append(("PyTorch fp16 @ (seq [1,N]@[N,M])", avg, mn, mx_t))

    # ====== 0b. Baseline: PyTorch native fp16 @ (one) ======
    avg, mn, mx_t = benchmark(lambda: pytorch_fp16_matmul_one(x_1d, w_fp16))
    results.append(("PyTorch fp16 @ (one [N]@[N,M])", avg, mn, mx_t))

    # ====== 1. rwkv pip cuBLAS gemm (seq) ======
    avg, mn, mx_t = benchmark(lambda: cublas_gemm_fp16_seq(x_2d, w_fp16))
    results.append(("rwkv pip cuBLAS gemm_fp16 (seq)", avg, mn, mx_t))

    # ====== 1b. rwkv pip cuBLAS gemm (one) ======
    avg, mn, mx_t = benchmark(lambda: cublas_gemm_fp16_one(x_1d, w_fp16))
    results.append(("rwkv pip cuBLAS gemm_fp16 (one)", avg, mn, mx_t))

    # ====== 2. rwkv pip PyTorch fallback (seq) ======
    avg, mn, mx_t = benchmark(lambda: rwkv_pip_torch_mm8_seq(x_2d, w_q, mx, rx, my, ry))
    results.append(("rwkv pip torch_mm8_seq", avg, mn, mx_t))

    # ====== 2b. rwkv pip PyTorch fallback (one) ======
    avg, mn, mx_t = benchmark(lambda: rwkv_pip_torch_mm8_one(x_1d, w_q, mx, rx, my, ry))
    results.append(("rwkv pip torch_mm8_one", avg, mn, mx_t))

    # ====== 3. rwkv pip CUDA kernel (seq, B=1) ======
    avg, mn, mx_t = benchmark(lambda: cuda_mm8_seq(1, N, M, x_2d, w_q, mx, rx, my, ry))
    results.append(("rwkv pip cuda_mm8_seq", avg, mn, mx_t))

    # ====== 3b. rwkv pip CUDA kernel (one) ======
    avg, mn, mx_t = benchmark(lambda: cuda_mm8_one(N, M, x_1d, w_q, mx, rx, my, ry))
    results.append(("rwkv pip cuda_mm8_one", avg, mn, mx_t))

    # ====== 4. Albatross PyTorch mm8 (seq, B=1) ======
    avg, mn, mx_t = benchmark(lambda: albatross_torch_mm8_seq(x_2d, w_q, mx, rx, my, ry))
    results.append(("Albatross torch_mm8_seq", avg, mn, mx_t))

    # ====== 4b. Albatross PyTorch mm8 (one) ======
    avg, mn, mx_t = benchmark(lambda: albatross_torch_mm8_one(x_1d, w_q, mx, rx, my, ry))
    results.append(("Albatross torch_mm8_one", avg, mn, mx_t))

    # ===================== Print latency results =====================
    print("-" * 90)
    print(f"{'Implementation':<45} {'Avg(μs)':>10} {'Min(μs)':>10} {'Max(μs)':>10}")
    print("-" * 90)
    for name, avg, mn, mx_t in results:
        print(f"{name:<45} {avg:>10.1f} {mn:>10.1f} {mx_t:>10.1f}")
    print("-" * 90)

    # Speedup relative to PyTorch fp16 @ (seq)
    baseline_avg = results[0][1]
    print(f"\n{'Implementation':<45} {'Speedup vs PyTorch fp16@':>25}")
    print("-" * 72)
    for name, avg, _, _ in results:
        speedup = baseline_avg / avg if avg > 0 else float('inf')
        print(f"{name:<45} {speedup:>21.2f}x")
    print("-" * 72)

    # ===================== Correctness check (vs fp16 ground truth) =====================
    def error_metrics(ref, y):
        """Compute max_abs_err, mean_abs_err, relative_l2_err, cosine_similarity."""
        diff = (ref - y).float()
        ref_f = ref.float()
        y_f = y.float()
        max_abs  = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        ref_norm = ref_f.norm().item()
        rel_l2   = diff.norm().item() / ref_norm if ref_norm > 0 else float('inf')
        cos_sim  = F.cosine_similarity(ref_f.view(1, -1), y_f.view(1, -1)).item()
        return max_abs, mean_abs, rel_l2, cos_sim

    ref_seq = (x_2d @ w_fp16)          # [1, M] fp16 ground truth
    ref_one = (x_1d @ w_fp16)          # [M]    fp16 ground truth

    checks = []
    # cuBLAS gemm
    checks.append(("cuBLAS gemm_fp16 (seq)", ref_seq, cublas_gemm_fp16_seq(x_2d, w_fp16)))
    checks.append(("cuBLAS gemm_fp16 (one)", ref_one, cublas_gemm_fp16_one(x_1d, w_fp16)))
    # rwkv pip torch fallback
    checks.append(("rwkv pip torch_mm8_seq", ref_seq, rwkv_pip_torch_mm8_seq(x_2d, w_q, mx, rx, my, ry)))
    checks.append(("rwkv pip torch_mm8_one", ref_one, rwkv_pip_torch_mm8_one(x_1d, w_q, mx, rx, my, ry)))
    # rwkv pip CUDA kernel
    checks.append(("rwkv pip cuda_mm8_seq",  ref_seq, cuda_mm8_seq(1, N, M, x_2d, w_q, mx, rx, my, ry)))
    checks.append(("rwkv pip cuda_mm8_one",  ref_one, cuda_mm8_one(N, M, x_1d, w_q, mx, rx, my, ry)))
    # Albatross torch mm8
    checks.append(("Albatross torch_mm8_seq", ref_seq, albatross_torch_mm8_seq(x_2d, w_q, mx, rx, my, ry)))
    checks.append(("Albatross torch_mm8_one", ref_one, albatross_torch_mm8_one(x_1d, w_q, mx, rx, my, ry)))

    print(f"\n{'=== Correctness vs fp16 ground truth (x @ w_fp16) ==='}")
    hdr = f"{'Implementation':<35} {'MaxAbs':>10} {'MeanAbs':>10} {'RelL2':>10} {'CosSim':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, ref, y in checks:
        ma, mea, rl2, cs = error_metrics(ref, y)
        print(f"{name:<35} {ma:>10.6f} {mea:>10.6f} {rl2:>10.6f} {cs:>10.8f}")


if __name__ == "__main__":
    main()
