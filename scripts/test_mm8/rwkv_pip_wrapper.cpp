#include <torch/extension.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

typedef at::Half fp16;

template <typename F>
void cuda_mm8_seq(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);
template <typename F>
void cuda_mm8_one(int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);

template <typename F>
void cuda_mm8_seq_opt(int B, int N, int M,
                      F *x, int x_stride,
                      uint8_t *w, int w_stride,
                      F *mx, F *rx,
                      F *my, F *ry,
                      F *y, int y_stride);

void launch_mm_seq_wmma(int B, int N, int M,
                        fp16 *x, int x_stride,
                        uint8_t *w, int w_stride,
                        fp16 *mx, fp16 *rx,
                        fp16 *my, fp16 *ry,
                        fp16 *y, int y_stride);

void launch_preprocess_fp16(int B, int N,
    fp16 *x, int x_stride, fp16 *ry, fp16 *my,
    fp16 *xs, int xs_stride, fp16 *xs_sum, fp16 *x_sum, fp16 *xmy_sum);

void launch_postprocess_fp16(int B, int M,
    fp16 *core, int core_stride, fp16 *rx, fp16 *mx,
    fp16 *xs_sum, fp16 *x_sum, fp16 *xmy_sum,
    fp16 *y, int y_stride);

void launch_cast_u8_to_fp16(int total, const uint8_t *src, fp16 *dst);
void launch_cast_u8_to_fp16_stream(int total, const uint8_t *src, fp16 *dst, cudaStream_t stream);

void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(1) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(1) == 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_mm8_seq(
            B, N, M,
            x.data_ptr<fp16>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<fp16>(), y.stride(0));
        break;
    case c10::ScalarType::Float:
        cuda_mm8_seq(
            B, N, M,
            x.data_ptr<float>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>(), y.stride(0));
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(0) == 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
        cuda_mm8_one(
            N, M,
            x.data_ptr<fp16>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<float>());
        break;
    case c10::ScalarType::Float:
        cuda_mm8_one(
            N, M,
            x.data_ptr<float>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>());
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

void gemm_fp16_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// Direct cuBLAS GEMM with raw pointers: C[m,n] = A[m,k] @ B[k,n]  (row-major fp16)
static inline void gemm_fp16_raw(int m, int n, int k,
                                  const fp16 *A, const fp16 *B, fp16 *C) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major trick: C = A*B  <==>  C^T = B^T * A^T  (column-major)
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k, &alpha,
        B, CUDA_R_16F, n,    // B^T in col-major = B in row-major
        A, CUDA_R_16F, k,    // A^T in col-major = A in row-major
        &beta,
        C, CUDA_R_16F, n,    // C^T in col-major = C in row-major
        CUDA_R_32F,
#if CUDA_VERSION >= 11000
        CUBLAS_GEMM_DEFAULT
#else
        CUBLAS_GEMM_DFALT_TENSOR_OP
#endif
    );
}

void mm8_seq_opt(int64_t B, int64_t N, int64_t M,
                 torch::Tensor &x, torch::Tensor &w,
                 torch::Tensor &mx, torch::Tensor &rx,
                 torch::Tensor &my, torch::Tensor &ry,
                 torch::Tensor &y) {
    assert(x.stride(1) == 1);
    assert(w.stride(1) == 1);
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    assert(y.stride(1) == 1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    switch (x.scalar_type()) {
    case c10::ScalarType::Half:
    {
        // Fused algebraic decomposition + cuBLAS Tensor Core path
        // Separate allocations (w_fp16 separate for better L2 behavior)
        auto small_buf = torch::empty({B * N + 3 * B + B * M}, x.options());
        auto w_fp16    = torch::empty({N, M}, x.options());

        fp16 *p           = small_buf.data_ptr<fp16>();
        fp16 *xs_ptr      = p;
        fp16 *xs_sum_ptr  = p + B * N;
        fp16 *x_sum_ptr   = xs_sum_ptr + B;
        fp16 *xmy_sum_ptr = x_sum_ptr + B;
        fp16 *core_ptr    = p + B * N + 3 * B;
        fp16 *w_fp16_ptr  = w_fp16.data_ptr<fp16>();

        // Fused pre-processing kernel: xs, xs_sum, x_sum, xmy_sum
        launch_preprocess_fp16(B, N,
            x.data_ptr<fp16>(), x.stride(0),
            ry.data_ptr<fp16>(), my.data_ptr<fp16>(),
            xs_ptr, N,
            xs_sum_ptr, x_sum_ptr, xmy_sum_ptr);

        // Cast w uint8 → fp16 (vectorized, 8 elements/thread)
        launch_cast_u8_to_fp16(N * M, w.data_ptr<uint8_t>(), w_fp16_ptr);

        // cuBLAS GEMM: core[B,M] = xs[B,N] @ w_fp16[N,M]  (raw pointers)
        gemm_fp16_raw(B, M, N, xs_ptr, w_fp16_ptr, core_ptr);

        // Fused post-processing kernel
        launch_postprocess_fp16(B, M,
            core_ptr, M,
            rx.data_ptr<fp16>(), mx.data_ptr<fp16>(),
            xs_sum_ptr, x_sum_ptr, xmy_sum_ptr,
            y.data_ptr<fp16>(), y.stride(0));
        break;
    }
    case c10::ScalarType::Float:
        cuda_mm8_seq_opt(
            B, N, M,
            x.data_ptr<float>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>(), y.stride(0));
        break;
    default:
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

TORCH_LIBRARY(rwkv_pip, m) {
    m.def("mm8_seq", mm8_seq);
    m.def("mm8_one", mm8_one);
    m.def("mm8_seq_opt", mm8_seq_opt);
    m.def("gemm_fp16_cublas", gemm_fp16_cublas);
}
