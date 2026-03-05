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
BSZ = 16       # batch size for 3D test
N_EMB = 23     # sequence / embedding dim for 3D test
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

# ======================= Implementation 3b: Optimized CUDA mm8_seq =======================
# N-dimension splitting + shared memory caching (see rwkv_pip_operators.cu)

def cuda_mm8_seq_opt(B, N, M, x, w, mx, rx, my, ry):
    assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
    assert x.dtype == torch.float32 or x.dtype == torch.float16
    assert w.dtype == torch.uint8
    assert x.shape == (B, N)
    assert w.shape == (N, M)
    assert rx.shape == mx.shape == (M,)
    assert ry.shape == my.shape == (N, 1)
    y = torch.empty((B, M), device=w.device, dtype=x.dtype)
    torch.ops.rwkv_pip.mm8_seq_opt(B, N, M, x, w, mx, rx, my, ry, y)
    return y

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

# ======================= Reshape wrappers (2D/3D transparent) =======================
# Each wrapper flattens leading dims to 2D, calls the underlying seq op, reshapes back.

def _flatten_leading(x):
    """If x is 3D+ [*, C], return (x_2d [B*T, C], original_shape). Else (x, None)."""
    if x.dim() > 2:
        return x.reshape(-1, x.shape[-1]), x.shape
    return x, None

def _unflatten(y, orig_shape):
    """Restore leading dims: [B*T, M] -> [*orig_leading, M]."""
    if orig_shape is not None:
        return y.reshape(*orig_shape[:-1], -1)
    return y

def pytorch_fp16_matmul_seq_3d(x, w_fp16):
    x_2d, shape = _flatten_leading(x)
    return _unflatten(x_2d @ w_fp16, shape)

def cublas_gemm_fp16_seq_3d(x, w_fp16):
    x_2d, shape = _flatten_leading(x)
    c = torch.empty((x_2d.shape[0], w_fp16.shape[-1]), dtype=x.dtype, device=x.device)
    torch.ops.rwkv_pip.gemm_fp16_cublas(x_2d, w_fp16, c)
    return _unflatten(c, shape)

def rwkv_pip_torch_mm8_seq_3d(x, w, mx, rx, my, ry):
    x_2d, shape = _flatten_leading(x)
    y = rwkv_pip_torch_mm8_seq(x_2d, w, mx, rx, my, ry)
    return _unflatten(y, shape)

def cuda_mm8_seq_3d(x, w, mx, rx, my, ry):
    """Wrapper: auto-compute B from x, handle 2D/3D."""
    x_2d, shape = _flatten_leading(x)
    B_flat = x_2d.shape[0]
    N_inner = x_2d.shape[1]
    M_out = w.shape[1]
    y = cuda_mm8_seq(B_flat, N_inner, M_out, x_2d, w, mx, rx, my, ry)
    return _unflatten(y, shape)

def cuda_mm8_seq_opt_3d(x, w, mx, rx, my, ry):
    """Wrapper: optimized cuda_mm8_seq, handle 2D/3D."""
    x_2d, shape = _flatten_leading(x)
    B_flat = x_2d.shape[0]
    N_inner = x_2d.shape[1]
    M_out = w.shape[1]
    y = cuda_mm8_seq_opt(B_flat, N_inner, M_out, x_2d, w, mx, rx, my, ry)
    return _unflatten(y, shape)

def albatross_torch_mm8_seq_3d(x, w, mx, rx, my, ry):
    x_2d, shape = _flatten_leading(x)
    y = albatross_torch_mm8_seq(x_2d, w, mx, rx, my, ry)
    return _unflatten(y, shape)

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
    print(f"mm8 benchmark:  x[{BSZ},{N_EMB},{N}] @ w[{N},{M}]   (fp16, CUDA)")
    print("=" * 80)

    torch.manual_seed(42)

    # ---- Create test data ----
    x_3d = torch.randn(BSZ, N_EMB, N, dtype=DTYPE, device=DEVICE)  # [BSZ, N_EMB, N]
    x_1d = torch.randn(N, dtype=DTYPE, device=DEVICE)              # [N]
    w_fp16 = torch.randn(N, M, dtype=DTYPE, device=DEVICE)         # [N, M]

    # Quantise
    w_q, mx, rx, my, ry = quantize_weight(w_fp16)
    w_q  = w_q.to(device=DEVICE).contiguous()
    mx   = mx.to(device=DEVICE)
    rx   = rx.to(device=DEVICE)
    my   = my.to(device=DEVICE)
    ry   = ry.to(device=DEVICE)

    print(f"\nShapes:  x_3d={list(x_3d.shape)}  x_1d={list(x_1d.shape)}")
    print(f"         w_fp16={list(w_fp16.shape)}  w_q={list(w_q.shape)} (uint8)")
    print(f"         mx={list(mx.shape)}  rx={list(rx.shape)}  my={list(my.shape)}  ry={list(ry.shape)}")
    print(f"\nWarmup={WARMUP}  Repeats={REPEATS}")
    print()

    seq_results = []
    one_results = []

    # ==================== SEQ benchmarks (3D: [BSZ, N_EMB, N] @ [N, M]) ====================

    avg, mn, mx_t = benchmark(lambda: pytorch_fp16_matmul_seq_3d(x_3d, w_fp16))
    seq_results.append(("PyTorch fp16 @ (seq)", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: cublas_gemm_fp16_seq_3d(x_3d, w_fp16))
    seq_results.append(("rwkv pip cuBLAS gemm_fp16 (seq)", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: rwkv_pip_torch_mm8_seq_3d(x_3d, w_q, mx, rx, my, ry))
    seq_results.append(("rwkv pip torch_mm8_seq", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: cuda_mm8_seq_3d(x_3d, w_q, mx, rx, my, ry))
    seq_results.append(("rwkv pip cuda_mm8_seq", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: cuda_mm8_seq_opt_3d(x_3d, w_q, mx, rx, my, ry))
    seq_results.append(("rwkv pip cuda_mm8_seq (optimized)", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: albatross_torch_mm8_seq_3d(x_3d, w_q, mx, rx, my, ry))
    seq_results.append(("Albatross torch_mm8_seq", avg, mn, mx_t))

    # ==================== ONE benchmarks (1D: [N] @ [N, M]) ====================

    avg, mn, mx_t = benchmark(lambda: pytorch_fp16_matmul_one(x_1d, w_fp16))
    one_results.append(("PyTorch fp16 @ (one)", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: cublas_gemm_fp16_one(x_1d, w_fp16))
    one_results.append(("rwkv pip cuBLAS gemm_fp16 (one)", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: rwkv_pip_torch_mm8_one(x_1d, w_q, mx, rx, my, ry))
    one_results.append(("rwkv pip torch_mm8_one", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: cuda_mm8_one(N, M, x_1d, w_q, mx, rx, my, ry))
    one_results.append(("rwkv pip cuda_mm8_one", avg, mn, mx_t))

    avg, mn, mx_t = benchmark(lambda: albatross_torch_mm8_one(x_1d, w_q, mx, rx, my, ry))
    one_results.append(("Albatross torch_mm8_one", avg, mn, mx_t))

    # ===================== Print latency results =====================

    def print_latency_table(title, results_list):
        print(f"\n{'--- ' + title + ' ---'}")
        print("-" * 90)
        print(f"{'Implementation':<45} {'Avg(μs)':>10} {'Min(μs)':>10} {'Max(μs)':>10}")
        print("-" * 90)
        for name, avg, mn, mx_t in results_list:
            print(f"{name:<45} {avg:>10.1f} {mn:>10.1f} {mx_t:>10.1f}")
        print("-" * 90)
        baseline_avg = results_list[0][1]
        print(f"\n{'Implementation':<45} {'Speedup vs baseline':>25}")
        print("-" * 72)
        for name, avg, _, _ in results_list:
            speedup = baseline_avg / avg if avg > 0 else float('inf')
            print(f"{name:<45} {speedup:>21.2f}x")
        print("-" * 72)

    print_latency_table(f"SEQ mode: x[{BSZ},{N_EMB},{N}] @ w[{N},{M}]", seq_results)
    print_latency_table(f"ONE mode: x[{N}] @ w[{N},{M}]", one_results)

    # ===================== Correctness check =====================
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

    def print_correctness_table(title, checks_list):
        print(f"\n{'=== Correctness: ' + title + ' ==='}")
        hdr = f"{'Implementation':<35} {'MaxAbs':>10} {'MeanAbs':>10} {'RelL2':>10} {'CosSim':>10}"
        print(hdr)
        print("-" * len(hdr))
        for name, ref, y in checks_list:
            ma, mea, rl2, cs = error_metrics(ref, y)
            print(f"{name:<35} {ma:>10.6f} {mea:>10.6f} {rl2:>10.6f} {cs:>10.8f}")

    ref_seq_fp16 = pytorch_fp16_matmul_seq_3d(x_3d, w_fp16)  # fp16 ground truth
    ref_one_fp16 = (x_1d @ w_fp16)

    torch_mm8_seq = rwkv_pip_torch_mm8_seq_3d(x_3d, w_q, mx, rx, my, ry)
    torch_mm8_one = rwkv_pip_torch_mm8_one(x_1d, w_q, mx, rx, my, ry)

    # --- Part 1: PyTorch fp16 vs rwkv pip torch_mm8 (quantization error baseline) ---
    print_correctness_table("SEQ: PyTorch fp16 vs torch_mm8", [
        ("rwkv pip torch_mm8", ref_seq_fp16, torch_mm8_seq),
    ])
    print_correctness_table("ONE: PyTorch fp16 vs torch_mm8", [
        ("rwkv pip torch_mm8", ref_one_fp16, torch_mm8_one),
    ])

    # --- Part 2: All implementations vs torch_mm8 baseline ---
    seq_checks = [
        ("PyTorch fp16 @",          torch_mm8_seq, ref_seq_fp16),
        ("cuBLAS gemm_fp16",        torch_mm8_seq, cublas_gemm_fp16_seq_3d(x_3d, w_fp16)),
        ("rwkv pip cuda_mm8",       torch_mm8_seq, cuda_mm8_seq_3d(x_3d, w_q, mx, rx, my, ry)),
        ("rwkv pip cuda_mm8 (opt)", torch_mm8_seq, cuda_mm8_seq_opt_3d(x_3d, w_q, mx, rx, my, ry)),
        ("Albatross torch_mm8",     torch_mm8_seq, albatross_torch_mm8_seq_3d(x_3d, w_q, mx, rx, my, ry)),
    ]
    one_checks = [
        ("PyTorch fp16 @",          torch_mm8_one, ref_one_fp16),
        ("cuBLAS gemm_fp16",        torch_mm8_one, cublas_gemm_fp16_one(x_1d, w_fp16)),
        ("rwkv pip cuda_mm8",       torch_mm8_one, cuda_mm8_one(N, M, x_1d, w_q, mx, rx, my, ry)),
        ("Albatross torch_mm8",     torch_mm8_one, albatross_torch_mm8_one(x_1d, w_q, mx, rx, my, ry)),
    ]

    print_correctness_table("SEQ vs torch_mm8 baseline", seq_checks)
    print_correctness_table("ONE vs torch_mm8 baseline", one_checks)


# ======================= RWKV7 Model Simulation Benchmark =======================

def model_simulation_benchmark():
    """
    Simulate RWKV7 7B model structure (4 layers + head).
    Only benchmarks quantizable operators (w8a16):
      - ATT: receptance.weight, key.weight, value.weight, output.weight  [4096, 4096] each
      - FFN: key.weight [4096, 16384], value.weight [16384, 4096] (transposed at load)
      - POST: head.weight [4096, 65536]
    Tracks per-operator VRAM peak overhead and computation speed at varied sequence lengths.
    Model total latency is estimated by summing per-operator timings (matmul-only, lower-bound).
    """
    print("\n" + "=" * 110)
    print("  RWKV7 7B Model Simulation Benchmark (w8a16)")
    print("=" * 110)

    # ==================== Config ====================
    N_LAYER_SIM = 4
    SEQ_LENS = [1, 4, 16, 64, 256, 1024]
    SIM_WARMUP = 30
    SIM_REPEATS = 100

    # Unique operator shapes:
    #   (name, N_in, N_out, count_per_layer, is_post)
    #
    # For F.linear(x, W) with W=[out, in]:  x @ W.T  →  mm8 weight = W.T = [in, out]
    # For x @ V (transposed at load):       x @ V    →  mm8 weight = V   = [in, out]
    OP_SPECS = [
        ("att.R/K/V/O", 4096,  4096,  4, False),   # 4 per layer (receptance, key, value, output)
        ("ffn.key",     4096,  16384, 1, False),     # 1 per layer: F.linear(k, K_) where K_=[16384,4096]
        ("ffn.value",   16384, 4096,  1, False),     # 1 per layer: k @ V_ where V_=[16384,4096] after .t()
        ("head",        4096,  65536, 0, True),      # 1 total (post): F.linear(x, head.weight)
    ]

    torch.manual_seed(42)

    # ==================== Create & quantize one weight per unique shape ====================
    print("\nAllocating model weights ...")
    op_weights = {}
    for sn, nin, nout, _, _ in OP_SPECS:
        wf = torch.randn(nin, nout, dtype=DTYPE, device=DEVICE)
        wq, mx_, rx_, my_, ry_ = quantize_weight(wf)
        op_weights[sn] = {
            'fp16': wf.contiguous(),
            'q':  wq.to(device=DEVICE).contiguous(),
            'mx': mx_.to(device=DEVICE).contiguous(),
            'rx': rx_.to(device=DEVICE).contiguous(),
            'my': my_.to(device=DEVICE).contiguous(),
            'ry': ry_.to(device=DEVICE).contiguous(),
        }

    # ==================== Weight memory summary ====================
    print(f"\nModel: {N_LAYER_SIM} layers  |  quantizable matmul operators only")
    print(f"  {'Operator':<18} {'Shape':<18} {'#Total':>7} {'fp16(MB)':>10} {'w8a16(MB)':>10}")
    print(f"  {'-' * 67}")
    t_fp16 = t_q = 0.0
    for sn, nin, nout, cnt, ip in OP_SPECS:
        tc = cnt * N_LAYER_SIM if not ip else 1
        fp_mb = nin * nout * 2 / 1024**2 * tc
        # quantized: uint8 weight + fp16 (mx[M], rx[M], my[N], ry[N])
        q_mb  = (nin * nout + 4 * (nin + nout)) / 1024**2 * tc
        t_fp16 += fp_mb; t_q += q_mb
        print(f"  {sn:<18} [{nin}x{nout}]{'':>5} {tc:>7} {fp_mb:>10.2f} {q_mb:>10.2f}")
    print(f"  {'-' * 67}")
    print(f"  {'TOTAL':<18} {'':>18} {'':>7} {t_fp16:>10.2f} {t_q:>10.2f}")
    print(f"  Compression: {t_fp16 / t_q:.2f}x  |  Saving: {(1 - t_q / t_fp16) * 100:.1f}%")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ==================== Implementation wrappers ====================
    # All wrappers: (x: [B, N_in], wd: dict) -> [B, N_out]

    def _fp16(x, wd):
        return x @ wd['fp16']

    def _alb_mm8(x, wd):
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        return albatross_torch_mm8_seq(x2, wd['q'], wd['mx'], wd['rx'], wd['my'], wd['ry'])

    def _pip_mm8(x, wd):
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        return rwkv_pip_torch_mm8_seq(x2, wd['q'], wd['mx'], wd['rx'], wd['my'], wd['ry'])

    def _cuda_mm8(x, wd):
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        B, Ni = x2.shape; Mo = wd['q'].shape[1]
        return cuda_mm8_seq(B, Ni, Mo, x2, wd['q'], wd['mx'], wd['rx'], wd['my'], wd['ry'])

    def _cuda_mm8o(x, wd):
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        B, Ni = x2.shape; Mo = wd['q'].shape[1]
        return cuda_mm8_seq_opt(B, Ni, Mo, x2, wd['q'], wd['mx'], wd['rx'], wd['my'], wd['ry'])

    IMPLS = [
        ("fp16 @",        _fp16),
        ("Albatross mm8", _alb_mm8),
        ("pip torch mm8", _pip_mm8),
        # ("CUDA mm8",      _cuda_mm8),
        ("CUDA mm8 opt",  _cuda_mm8o),
    ]

    # ==================== Single-op benchmark utility ====================

    def _bench(fn, warmup=SIM_WARMUP, reps=SIM_REPEATS):
        """Return (vram_peak_overhead_MB, avg_latency_us)."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # --- VRAM peak overhead (activation + temporaries) ---
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        vram = (torch.cuda.max_memory_allocated() - base) / 1024**2

        # --- Latency (CUDA events, trimmed mean) ---
        se = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ee = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            se[i].record(); fn(); ee[i].record()
        torch.cuda.synchronize()
        ts = sorted(s.elapsed_time(e) * 1000 for s, e in zip(se, ee))  # ms -> us
        t = max(1, reps // 10)
        return vram, sum(ts[t:-t]) / len(ts[t:-t])

    # ==================== Run all benchmarks ====================
    # R[seq_len][impl_name][op_name] = (vram_mb, latency_us)
    R = {}
    for sl in SEQ_LENS:
        R[sl] = {}
        print(f"\n  Benchmarking seq_len={sl} ...", end="", flush=True)
        for iname, ifn in IMPLS:
            R[sl][iname] = {}
            for sn, nin, nout, _, _ in OP_SPECS:
                x = torch.randn(sl, nin, dtype=DTYPE, device=DEVICE)
                wd = op_weights[sn]
                vram, lat = _bench(lambda _f=ifn, _x=x, _w=wd: _f(_x, _w))
                R[sl][iname][sn] = (vram, lat)
                del x
                torch.cuda.empty_cache()
            print(f"  [{iname}]", end="", flush=True)
        print("  done.")

    # ==================== Detailed results per seq_len ====================
    for sl in SEQ_LENS:
        print(f"\n{'=' * 110}")
        print(f"  seq_len = {sl}")
        print(f"{'=' * 110}")

        for sn, nin, nout, cnt, ip in OP_SPECS:
            tc = cnt * N_LAYER_SIM if not ip else 1
            print(f"\n  {sn} [{nin}x{nout}]  x{tc} in model")
            print(f"  {'Implementation':<22} {'VRAM peak(MB)':>13} {'Latency(us)':>13} {'ModelTotal(us)':>15} {'vs fp16':>8}")
            print(f"  {'-' * 75}")
            base_lat = R[sl][IMPLS[0][0]][sn][1]
            for iname, _ in IMPLS:
                v, l = R[sl][iname][sn]
                tl = l * tc
                sp = base_lat / l if l > 0 else float('inf')
                print(f"  {iname:<22} {v:>13.2f} {l:>13.1f} {tl:>15.1f} {sp:>7.2f}x")

        # Full model estimate
        print(f"\n  --- Full model estimate ({N_LAYER_SIM} layers + head, matmul-only) ---")
        print(f"  {'Implementation':<22} {'TotalLatency(us)':>17} {'MaxVRAMpeak(MB)':>17} {'vs fp16':>8}")
        print(f"  {'-' * 68}")
        base_total = None
        for iname, _ in IMPLS:
            tl = sum(R[sl][iname][sn][1] * (c * N_LAYER_SIM if not ip else 1)
                     for sn, _, _, c, ip in OP_SPECS)
            mv = max(R[sl][iname][sn][0] for sn, _, _, _, _ in OP_SPECS)
            if base_total is None:
                base_total = tl
            sp = base_total / tl if tl > 0 else float('inf')
            print(f"  {iname:<22} {tl:>17.1f} {mv:>17.2f} {sp:>7.2f}x")

    # ==================== Summary across all seq_lens ====================
    print(f"\n{'=' * 110}")
    print(f"  Summary: Estimated Total Model Latency (us)")
    print(f"{'=' * 110}")
    hdr = f"  {'Implementation':<22}"
    for sl in SEQ_LENS:
        hdr += f"  {'seq=' + str(sl):>10}"
    print(hdr)
    print(f"  {'-' * (22 + 12 * len(SEQ_LENS))}")
    for iname, _ in IMPLS:
        row = f"  {iname:<22}"
        for sl in SEQ_LENS:
            tl = sum(R[sl][iname][sn][1] * (c * N_LAYER_SIM if not ip else 1)
                     for sn, _, _, c, ip in OP_SPECS)
            row += f"  {tl:>10.1f}"
        print(row)

    print(f"\n  Summary: Max VRAM Peak Overhead (MB)")
    hdr = f"  {'Implementation':<22}"
    for sl in SEQ_LENS:
        hdr += f"  {'seq=' + str(sl):>10}"
    print(hdr)
    print(f"  {'-' * (22 + 12 * len(SEQ_LENS))}")
    for iname, _ in IMPLS:
        row = f"  {iname:<22}"
        for sl in SEQ_LENS:
            mv = max(R[sl][iname][sn][0] for sn, _, _, _, _ in OP_SPECS)
            row += f"  {mv:>10.2f}"
        print(row)

    print()


if __name__ == "__main__":
    main()
    model_simulation_benchmark()
