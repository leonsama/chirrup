#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>

typedef at::Half fp16;

__half *cast(fp16 *ptr) {
    return reinterpret_cast<__half *>(ptr);
}

__global__ void kernel_mm_seq_fp32i8(
    const int B, const int N, const int M,
    const float *__restrict__ const x, const int x_stride,
    const uint8_t *__restrict__ const w, const int w_stride,
    const float *__restrict__ const mx,
    const float *__restrict__ const rx,
    const float *__restrict__ const my,
    const float *__restrict__ const ry,
    float *__restrict__ const y, const int y_stride) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && k < M) {
        float y_local = 0;
        for (int j = 0; j < N; ++j) {
            y_local += x[i * x_stride + j] * (
                (float(w[j * w_stride + k]) + 0.5f)
                * rx[k] * ry[j] + mx[k] + my[j]
            );
        }
        y[i * y_stride + k] = y_local;
    }
}

template <typename F>
void cuda_mm8_seq(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);

template <>
void cuda_mm8_seq<float>(int B, int N, int M,
                         float *x, int x_stride,
                         uint8_t *w, int w_stride,
                         float *mx, float *rx,
                         float *my, float *ry,
                         float *y, int y_stride) {
    dim3 blockSize(1, 128);
    dim3 gridSize((B + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm_seq_fp32i8<<<gridSize, blockSize>>>(
        B, N, M, x, x_stride, w, w_stride,
        mx, rx, my, ry, y, y_stride);
}

__global__ void kernel_mm_seq_fp16i8(
    const int B, const int N, const int M,
    const __half *__restrict__ const x, const int x_stride,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    __half *__restrict__ const y, const int y_stride) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && k < M) {
        float y_local = 0;
        for (int j = 0; j < N; ++j) {
            y_local += __half2float(x[i * x_stride + j]) * (
                (float(w[j * w_stride + k]) + 0.5f)
                * __half2float(rx[k]) * __half2float(ry[j])
                + __half2float(mx[k]) + __half2float(my[j])
            );
        }
        y[i * y_stride + k] = __float2half(y_local);
    }
}

template <>
void cuda_mm8_seq<fp16>(int B, int N, int M,
                        fp16 *x, int x_stride,
                        uint8_t *w, int w_stride,
                        fp16 *mx, fp16 *rx,
                        fp16 *my, fp16 *ry,
                        fp16 *y, int y_stride) {
    dim3 blockSize(1, 128);
    dim3 gridSize((B + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm_seq_fp16i8<<<gridSize, blockSize>>>(
        B, N, M, cast(x), x_stride, w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), cast(y), y_stride);
}

#define MM8_ONE_JSPLIT 24
#define MM8_ONE_TILE 1024

__global__ void kernel_mm_one_fp32i8(
    const int N, const int M,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    const float *__restrict__ const mx,
    const float *__restrict__ const rx,
    const float *__restrict__ const my,
    const float *__restrict__ const ry,
    float *__restrict__ const y) {

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M) {
        float y_local = 0;
        for (int j = j0; j < j1; ++j) {
            y_local += x[j] * (
                (float(w[j * w_stride + k]) + 0.5f)
                * rx[k] * ry[j] + mx[k] + my[j]
            );
        }
        atomicAdd(&y[k], y_local);
    }
}

template <typename F>
void cuda_mm8_one(int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);

template <>
void cuda_mm8_one<float>(int N, int M,
                         float *x,
                         uint8_t *w, int w_stride,
                         float *mx, float *rx,
                         float *my, float *ry,
                         float *y) {
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm_one_fp32i8<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride,
        mx, rx, my, ry, y);
}

__global__ void kernel_mm_one_fp16i8(
    const int N, const int M,
    const __half *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    float *__restrict__ const y) {

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M) {
        float y_local = 0;
        for (int j = j0; j < j1; ++j) {
            y_local += __half2float(x[j]) * (
                (float(w[j * w_stride + k]) + 0.5f)
                * __half2float(rx[k]) * __half2float(ry[j])
                + __half2float(mx[k]) + __half2float(my[j])
            );
        }
        atomicAdd(&y[k], y_local);
    }
}

template <>
void cuda_mm8_one<fp16>(int N, int M,
                        fp16 *x,
                        uint8_t *w, int w_stride,
                        fp16 *mx, fp16 *rx,
                        fp16 *my, fp16 *ry,
                        float *y) {
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm_one_fp16i8<<<gridSize, blockSize>>>(
        N, M, cast(x), w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), y);
}

// ======================= Optimized mm8_seq with batch + N tiling =======================
// Key optimizations vs original kernel_mm_seq_fp16i8:
// 1. Batch tiling: BTILE batch elements share same w reads (8x w bandwidth reduction)
// 2. Cooperative loading: all 256 threads load sx[8][32] in one pass
//    (BTILE * JTILE = 256 = KTILE, so each thread loads exactly one element)
// 3. Zero-fill OOB batch elements → no branch divergence in inner loop
// 4. #pragma unroll on all inner loops for ILP
// 5. Dequantized value computed once per (j,k), reused across BTILE batch elements

#define MM8_SEQ_OPT_BTILE 32    // batch elements per block
#define MM8_SEQ_OPT_JTILE 32    // inner loop (N) tile
#define MM8_SEQ_OPT_KTILE 256   // output columns per block (= blockDim.x)
// Cooperative loading: each thread loads BTILE*JTILE/KTILE elements per tile

__global__ void kernel_mm_seq_fp32i8_opt(
    const int B, const int N, const int M,
    const float *__restrict__ const x, const int x_stride,
    const uint8_t *__restrict__ const w, const int w_stride,
    const float *__restrict__ const mx,
    const float *__restrict__ const rx,
    const float *__restrict__ const my,
    const float *__restrict__ const ry,
    float *__restrict__ const y, const int y_stride) {

    const int k = blockIdx.x * MM8_SEQ_OPT_KTILE + threadIdx.x;
    const int i_base = blockIdx.y * MM8_SEQ_OPT_BTILE;

    float acc[MM8_SEQ_OPT_BTILE];
    #pragma unroll
    for (int ii = 0; ii < MM8_SEQ_OPT_BTILE; ++ii) acc[ii] = 0.0f;

    float rx_k = 0, mx_k = 0;
    if (k < M) { rx_k = rx[k]; mx_k = mx[k]; }

    __shared__ float sx[MM8_SEQ_OPT_BTILE][MM8_SEQ_OPT_JTILE];
    __shared__ float sry[MM8_SEQ_OPT_JTILE];
    __shared__ float smy[MM8_SEQ_OPT_JTILE];

    const int N_full = (N / MM8_SEQ_OPT_JTILE) * MM8_SEQ_OPT_JTILE;

    for (int j_base = 0; j_base < N_full; j_base += MM8_SEQ_OPT_JTILE) {
        // Cooperative load: BTILE*JTILE elements, KTILE threads
        for (int flat = threadIdx.x; flat < MM8_SEQ_OPT_BTILE * MM8_SEQ_OPT_JTILE; flat += MM8_SEQ_OPT_KTILE) {
            const int ii = flat / MM8_SEQ_OPT_JTILE;
            const int jj = flat % MM8_SEQ_OPT_JTILE;
            const int i = i_base + ii;
            sx[ii][jj] = (i < B) ? x[i * x_stride + j_base + jj] : 0.0f;
        }
        if (threadIdx.x < MM8_SEQ_OPT_JTILE) {
            sry[threadIdx.x] = ry[j_base + threadIdx.x];
            smy[threadIdx.x] = my[j_base + threadIdx.x];
        }
        __syncthreads();

        if (k < M) {
            #pragma unroll
            for (int jj = 0; jj < MM8_SEQ_OPT_JTILE; ++jj) {
                float dequant = (float(w[(j_base + jj) * w_stride + k]) + 0.5f)
                    * rx_k * sry[jj] + mx_k + smy[jj];
                #pragma unroll
                for (int ii = 0; ii < MM8_SEQ_OPT_BTILE; ++ii) {
                    acc[ii] += sx[ii][jj] * dequant;
                }
            }
        }
        __syncthreads();
    }

    // Handle remainder tile (if N % JTILE != 0)
    if (N_full < N) {
        const int j_len = N - N_full;
        for (int flat = threadIdx.x; flat < MM8_SEQ_OPT_BTILE * MM8_SEQ_OPT_JTILE; flat += MM8_SEQ_OPT_KTILE) {
            const int ii = flat / MM8_SEQ_OPT_JTILE;
            const int jj = flat % MM8_SEQ_OPT_JTILE;
            const int i = i_base + ii;
            sx[ii][jj] = (i < B && jj < j_len) ? x[i * x_stride + N_full + jj] : 0.0f;
        }
        if (threadIdx.x < (unsigned)j_len) {
            sry[threadIdx.x] = ry[N_full + threadIdx.x];
            smy[threadIdx.x] = my[N_full + threadIdx.x];
        }
        __syncthreads();

        if (k < M) {
            for (int jj = 0; jj < j_len; ++jj) {
                float dequant = (float(w[(N_full + jj) * w_stride + k]) + 0.5f)
                    * rx_k * sry[jj] + mx_k + smy[jj];
                #pragma unroll
                for (int ii = 0; ii < MM8_SEQ_OPT_BTILE; ++ii) {
                    acc[ii] += sx[ii][jj] * dequant;
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if (k < M) {
        #pragma unroll
        for (int ii = 0; ii < MM8_SEQ_OPT_BTILE; ++ii) {
            int i = i_base + ii;
            if (i < B) y[i * y_stride + k] = acc[ii];
        }
    }
}

__global__ void kernel_mm_seq_fp16i8_opt(
    const int B, const int N, const int M,
    const __half *__restrict__ const x, const int x_stride,
    const uint8_t *__restrict__ const w, const int w_stride,
    const __half *__restrict__ const mx,
    const __half *__restrict__ const rx,
    const __half *__restrict__ const my,
    const __half *__restrict__ const ry,
    __half *__restrict__ const y, const int y_stride) {

    const int k = blockIdx.x * MM8_SEQ_OPT_KTILE + threadIdx.x;
    const int i_base = blockIdx.y * MM8_SEQ_OPT_BTILE;

    float acc[MM8_SEQ_OPT_BTILE];
    #pragma unroll
    for (int ii = 0; ii < MM8_SEQ_OPT_BTILE; ++ii) acc[ii] = 0.0f;

    float rx_k = 0, mx_k = 0;
    if (k < M) { rx_k = __half2float(rx[k]); mx_k = __half2float(mx[k]); }

    // Pad shared memory to avoid bank conflicts
    __shared__ float sx[MM8_SEQ_OPT_BTILE][MM8_SEQ_OPT_JTILE];
    __shared__ float sry[MM8_SEQ_OPT_JTILE];
    __shared__ float smy[MM8_SEQ_OPT_JTILE];

    const int N_full = (N / MM8_SEQ_OPT_JTILE) * MM8_SEQ_OPT_JTILE;

    for (int j_base = 0; j_base < N_full; j_base += MM8_SEQ_OPT_JTILE) {
        for (int flat = threadIdx.x; flat < MM8_SEQ_OPT_BTILE * MM8_SEQ_OPT_JTILE; flat += MM8_SEQ_OPT_KTILE) {
            const int ii = flat / MM8_SEQ_OPT_JTILE;
            const int jj = flat % MM8_SEQ_OPT_JTILE;
            const int i = i_base + ii;
            sx[ii][jj] = (i < B) ? __half2float(x[i * x_stride + j_base + jj]) : 0.0f;
        }
        if (threadIdx.x < MM8_SEQ_OPT_JTILE) {
            sry[threadIdx.x] = __half2float(ry[j_base + threadIdx.x]);
            smy[threadIdx.x] = __half2float(my[j_base + threadIdx.x]);
        }
        __syncthreads();

        if (k < M) {
            #pragma unroll
            for (int jj = 0; jj < MM8_SEQ_OPT_JTILE; ++jj) {
                float dequant = (float(w[(j_base + jj) * w_stride + k]) + 0.5f)
                    * rx_k * sry[jj] + mx_k + smy[jj];
                #pragma unroll
                for (int ii = 0; ii < MM8_SEQ_OPT_BTILE; ++ii) {
                    acc[ii] += sx[ii][jj] * dequant;
                }
            }
        }
        __syncthreads();
    }

    if (N_full < N) {
        const int j_len = N - N_full;
        for (int flat = threadIdx.x; flat < MM8_SEQ_OPT_BTILE * MM8_SEQ_OPT_JTILE; flat += MM8_SEQ_OPT_KTILE) {
            const int ii = flat / MM8_SEQ_OPT_JTILE;
            const int jj = flat % MM8_SEQ_OPT_JTILE;
            const int i = i_base + ii;
            sx[ii][jj] = (i < B && jj < j_len) ? __half2float(x[i * x_stride + N_full + jj]) : 0.0f;
        }
        if (threadIdx.x < (unsigned)j_len) {
            sry[threadIdx.x] = __half2float(ry[N_full + threadIdx.x]);
            smy[threadIdx.x] = __half2float(my[N_full + threadIdx.x]);
        }
        __syncthreads();

        if (k < M) {
            for (int jj = 0; jj < j_len; ++jj) {
                float dequant = (float(w[(N_full + jj) * w_stride + k]) + 0.5f)
                    * rx_k * sry[jj] + mx_k + smy[jj];
                #pragma unroll
                for (int ii = 0; ii < MM8_SEQ_OPT_BTILE; ++ii) {
                    acc[ii] += sx[ii][jj] * dequant;
                }
            }
        }
        __syncthreads();
    }

    if (k < M) {
        #pragma unroll
        for (int ii = 0; ii < MM8_SEQ_OPT_BTILE; ++ii) {
            int i = i_base + ii;
            if (i < B) y[i * y_stride + k] = __float2half(acc[ii]);
        }
    }
}

template <typename F>
void cuda_mm8_seq_opt(int B, int N, int M,
                      F *x, int x_stride,
                      uint8_t *w, int w_stride,
                      F *mx, F *rx,
                      F *my, F *ry,
                      F *y, int y_stride);

template <>
void cuda_mm8_seq_opt<float>(int B, int N, int M,
                             float *x, int x_stride,
                             uint8_t *w, int w_stride,
                             float *mx, float *rx,
                             float *my, float *ry,
                             float *y, int y_stride) {
    dim3 blockSize(MM8_SEQ_OPT_KTILE);
    dim3 gridSize((M + MM8_SEQ_OPT_KTILE - 1) / MM8_SEQ_OPT_KTILE,
                  (B + MM8_SEQ_OPT_BTILE - 1) / MM8_SEQ_OPT_BTILE);
    int smem_bytes = (MM8_SEQ_OPT_BTILE + 2) * MM8_SEQ_OPT_JTILE * sizeof(float);
    kernel_mm_seq_fp32i8_opt<<<gridSize, blockSize, smem_bytes>>>(
        B, N, M, x, x_stride, w, w_stride,
        mx, rx, my, ry, y, y_stride);
}

template <>
void cuda_mm8_seq_opt<fp16>(int B, int N, int M,
                            fp16 *x, int x_stride,
                            uint8_t *w, int w_stride,
                            fp16 *mx, fp16 *rx,
                            fp16 *my, fp16 *ry,
                            fp16 *y, int y_stride) {
    dim3 blockSize(MM8_SEQ_OPT_KTILE);
    dim3 gridSize((M + MM8_SEQ_OPT_KTILE - 1) / MM8_SEQ_OPT_KTILE,
                  (B + MM8_SEQ_OPT_BTILE - 1) / MM8_SEQ_OPT_BTILE);
    int smem_bytes = (MM8_SEQ_OPT_BTILE + 2) * MM8_SEQ_OPT_JTILE * sizeof(float);
    kernel_mm_seq_fp16i8_opt<<<gridSize, blockSize, smem_bytes>>>(
        B, N, M, cast(x), x_stride, w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry), cast(y), y_stride);
}

// ======================= WMMA Tensor-Core mm8_seq =======================
// Fused dequantization + WMMA GEMM. Single kernel, same API as scalar version.
// Dequantizes w in shared memory, then uses Tensor Cores for the matmul.

#include <mma.h>
using namespace nvcuda;

#define TC_BM 64
#define TC_BN 64
#define TC_BK 64
#define TC_WARP_ROWS (TC_BM / 16)   // 4
#define TC_WARP_COLS (TC_BN / 16)   // 4
#define TC_NWARPS    (TC_WARP_ROWS * TC_WARP_COLS)  // 16
#define TC_THREADS   (TC_NWARPS * 32)               // 512
// Dynamic smem layout (reuse sa/sb space for sc after loop):
// Phase 1 (loop):  sa[BM*BK] fp16 + sb[BK*BN] fp16 + s_rx/mx[BN] + s_ry/my[BK] float
// Phase 2 (store): sc[BM*BN] float  (overlaps sa/sb)
#define TC_SMEM_TILED (TC_BM*TC_BK*2 + TC_BK*TC_BN*2 + (2*TC_BN + 2*TC_BK)*4)
#define TC_SMEM_OUT   (TC_BM*TC_BN*4)
#define TC_SMEM_BYTES (TC_SMEM_TILED > TC_SMEM_OUT ? TC_SMEM_TILED : TC_SMEM_OUT)

__global__ void kernel_mm_seq_wmma_fused(
    const int B, const int N, const int M,
    const __half *__restrict__ x, const int x_stride,
    const uint8_t *__restrict__ w, const int w_stride,
    const __half *__restrict__ mx,
    const __half *__restrict__ rx,
    const __half *__restrict__ my,
    const __half *__restrict__ ry,
    __half *__restrict__ y, const int y_stride)
{
    extern __shared__ char smem_raw[];

    // Tiled-phase pointers
    __half *sa  = reinterpret_cast<__half*>(smem_raw);                             // [BM, BK]
    __half *sb  = sa + TC_BM * TC_BK;                                              // [BK, BN]
    float  *s_rx = reinterpret_cast<float*>(sb + TC_BK * TC_BN);                  // [BN]
    float  *s_mx = s_rx + TC_BN;                                                   // [BN]
    float  *s_ry = s_mx + TC_BN;                                                   // [BK]
    float  *s_my = s_ry + TC_BK;                                                   // [BK]

    const int bm = blockIdx.y * TC_BM;
    const int bn = blockIdx.x * TC_BN;
    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id / TC_WARP_COLS;
    const int warp_col = warp_id % TC_WARP_COLS;

    // Load rx, mx for this output column block (once)
    for (int f = threadIdx.x; f < TC_BN; f += TC_THREADS) {
        int gk = bn + f;
        s_rx[f] = (gk < M) ? __half2float(rx[gk]) : 0.0f;
        s_mx[f] = (gk < M) ? __half2float(mx[gk]) : 0.0f;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> C;
    wmma::fill_fragment(C, 0.0f);

    for (int k = 0; k < N; k += TC_BK) {
        const int k_rem = min(TC_BK, N - k);

        // Load ry, my for this K-tile
        if (threadIdx.x < (unsigned)TC_BK) {
            int gj = k + threadIdx.x;
            s_ry[threadIdx.x] = (gj < N) ? __half2float(ry[gj]) : 0.0f;
            s_my[threadIdx.x] = (gj < N) ? __half2float(my[gj]) : 0.0f;
        }

        // Load x tile [TC_BM, TC_BK] into sa
        for (int f = threadIdx.x; f < TC_BM * TC_BK; f += TC_THREADS) {
            int r = f / TC_BK, c = f % TC_BK;
            sa[r * TC_BK + c] = (bm + r < B && c < k_rem)
                ? x[(bm + r) * x_stride + k + c]
                : __float2half(0.0f);
        }
        __syncthreads();  // Ensure s_ry/s_my and sa ready

        // Dequantize w tile [TC_BK, TC_BN] and store as fp16 in sb
        for (int f = threadIdx.x; f < TC_BK * TC_BN; f += TC_THREADS) {
            int j = f / TC_BN, kk = f % TC_BN;
            if (j < k_rem && bn + kk < M) {
                float dq = ((float)w[(k + j) * w_stride + bn + kk] + 0.5f)
                    * s_rx[kk] * s_ry[j] + s_mx[kk] + s_my[j];
                sb[j * TC_BN + kk] = __float2half(dq);
            } else {
                sb[j * TC_BN + kk] = __float2half(0.0f);
            }
        }
        __syncthreads();  // Ensure sb ready

        // WMMA: TC_BK/16 steps per K-tile
        #pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> A;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> B_frag;
            wmma::load_matrix_sync(A, &sa[warp_row * 16 * TC_BK + kk], TC_BK);
            wmma::load_matrix_sync(B_frag, &sb[kk * TC_BN + warp_col * 16], TC_BN);
            wmma::mma_sync(C, A, B_frag, C);
        }

        __syncthreads();  // Ensure WMMA reads done before next tile overwrites sa/sb
    }

    // Store: reuse smem space for sc[TC_BM][TC_BN] float
    float *sc = reinterpret_cast<float*>(smem_raw);  // overlaps sa/sb (no longer needed)
    wmma::store_matrix_sync(&sc[warp_row * 16 * TC_BN + warp_col * 16], C, TC_BN,
                            wmma::mem_row_major);
    __syncthreads();

    for (int f = threadIdx.x; f < TC_BM * TC_BN; f += TC_THREADS) {
        int r = f / TC_BN, c = f % TC_BN;
        if (bm + r < B && bn + c < M)
            y[(bm + r) * y_stride + bn + c] = __float2half(sc[r * TC_BN + c]);
    }
}

void launch_mm_seq_wmma(
    int B, int N, int M,
    fp16 *x, int x_stride,
    uint8_t *w, int w_stride,
    fp16 *mx, fp16 *rx,
    fp16 *my, fp16 *ry,
    fp16 *y, int y_stride)
{
    dim3 grid((M + TC_BN - 1) / TC_BN, (B + TC_BM - 1) / TC_BM);
    dim3 block(TC_THREADS);
    int smem_bytes = TC_SMEM_BYTES;
    kernel_mm_seq_wmma_fused<<<grid, block, smem_bytes>>>(
        B, N, M,
        cast(x), x_stride, w, w_stride,
        cast(mx), cast(rx), cast(my), cast(ry),
        cast(y), y_stride);
}

// ======================= Fused pre-processing kernel =======================
// Computes: xs[i,j] = x[i,j] * ry[j]
//           xs_sum[i], x_sum[i], xmy_sum[i] (reductions over j)

__global__ void kernel_preprocess_fp16(
    const int B, const int N,
    const __half *__restrict__ x, const int x_stride,
    const __half *__restrict__ ry,
    const __half *__restrict__ my,
    __half *__restrict__ xs, const int xs_stride,
    __half *__restrict__ xs_sum,
    __half *__restrict__ x_sum,
    __half *__restrict__ xmy_sum)
{
    const int i = blockIdx.x;
    if (i >= B) return;

    float local_xs_sum = 0.0f, local_x_sum = 0.0f, local_xmy_sum = 0.0f;

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float xv = __half2float(x[i * x_stride + j]);
        float ryv = __half2float(ry[j]);
        float myv = __half2float(my[j]);
        float xsv = xv * ryv;
        xs[i * xs_stride + j] = __float2half(xsv);
        local_xs_sum += xsv;
        local_x_sum += xv;
        local_xmy_sum += xv * myv;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_xs_sum  += __shfl_down_sync(0xFFFFFFFF, local_xs_sum, offset);
        local_x_sum   += __shfl_down_sync(0xFFFFFFFF, local_x_sum, offset);
        local_xmy_sum += __shfl_down_sync(0xFFFFFFFF, local_xmy_sum, offset);
    }

    __shared__ float s_xs[32], s_x[32], s_xmy[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) { s_xs[warp_id] = local_xs_sum; s_x[warp_id] = local_x_sum; s_xmy[warp_id] = local_xmy_sum; }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (threadIdx.x < 32) {
        float v_xs = (threadIdx.x < (unsigned)num_warps) ? s_xs[threadIdx.x] : 0.0f;
        float v_x = (threadIdx.x < (unsigned)num_warps) ? s_x[threadIdx.x] : 0.0f;
        float v_xmy = (threadIdx.x < (unsigned)num_warps) ? s_xmy[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v_xs  += __shfl_down_sync(0xFFFFFFFF, v_xs, offset);
            v_x   += __shfl_down_sync(0xFFFFFFFF, v_x, offset);
            v_xmy += __shfl_down_sync(0xFFFFFFFF, v_xmy, offset);
        }
        if (threadIdx.x == 0) {
            xs_sum[i] = __float2half(v_xs);
            x_sum[i] = __float2half(v_x);
            xmy_sum[i] = __float2half(v_xmy);
        }
    }
}

// ======================= Fused post-processing kernel =======================
// y[i,k] = rx[k] * (core[i,k] + 0.5 * xs_sum[i]) + xmy_sum[i] + mx[k] * x_sum[i]

__global__ void kernel_postprocess_fp16(
    const int B, const int M,
    const __half *__restrict__ core, const int core_stride,
    const __half *__restrict__ rx,
    const __half *__restrict__ mx,
    const __half *__restrict__ xs_sum,
    const __half *__restrict__ x_sum,
    const __half *__restrict__ xmy_sum,
    __half *__restrict__ y, const int y_stride)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (i >= B || k >= M) return;

    float cv = __half2float(core[i * core_stride + k]);
    float rxk = __half2float(rx[k]);
    float mxk = __half2float(mx[k]);
    float xs_s = __half2float(xs_sum[i]);
    float x_s = __half2float(x_sum[i]);
    float xmy_s = __half2float(xmy_sum[i]);

    y[i * y_stride + k] = __float2half(rxk * (cv + 0.5f * xs_s) + xmy_s + mxk * x_s);
}

// ======================= Fast uint8 → fp16 cast kernel =======================

__global__ void kernel_cast_u8_to_fp16(
    const int total,
    const uint8_t *__restrict__ src,
    __half *__restrict__ dst)
{
    // 8 elements per thread using uint2 (64-bit) load + half2 stores
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < total) {
        uint2 packed = *reinterpret_cast<const uint2*>(&src[idx]);
        *reinterpret_cast<__half2*>(&dst[idx + 0]) = __halves2half2(
            __float2half((float)(packed.x & 0xFF)),
            __float2half((float)((packed.x >> 8) & 0xFF)));
        *reinterpret_cast<__half2*>(&dst[idx + 2]) = __halves2half2(
            __float2half((float)((packed.x >> 16) & 0xFF)),
            __float2half((float)((packed.x >> 24) & 0xFF)));
        *reinterpret_cast<__half2*>(&dst[idx + 4]) = __halves2half2(
            __float2half((float)(packed.y & 0xFF)),
            __float2half((float)((packed.y >> 8) & 0xFF)));
        *reinterpret_cast<__half2*>(&dst[idx + 6]) = __halves2half2(
            __float2half((float)((packed.y >> 16) & 0xFF)),
            __float2half((float)((packed.y >> 24) & 0xFF)));
    } else {
        for (int d = 0; d < 8 && idx + d < total; ++d)
            dst[idx + d] = __float2half((float)src[idx + d]);
    }
}

void launch_preprocess_fp16(int B, int N,
    fp16 *x, int x_stride, fp16 *ry, fp16 *my,
    fp16 *xs, int xs_stride, fp16 *xs_sum, fp16 *x_sum, fp16 *xmy_sum)
{
    kernel_preprocess_fp16<<<B, min(1024, N)>>>(
        B, N, cast(x), x_stride, cast(ry), cast(my),
        cast(xs), xs_stride, cast(xs_sum), cast(x_sum), cast(xmy_sum));
}

void launch_postprocess_fp16(int B, int M,
    fp16 *core, int core_stride, fp16 *rx, fp16 *mx,
    fp16 *xs_sum, fp16 *x_sum, fp16 *xmy_sum,
    fp16 *y, int y_stride)
{
    dim3 block(256);
    dim3 grid((M + 255) / 256, B);
    kernel_postprocess_fp16<<<grid, block>>>(
        B, M, cast(core), core_stride, cast(rx), cast(mx),
        cast(xs_sum), cast(x_sum), cast(xmy_sum), cast(y), y_stride);
}

void launch_cast_u8_to_fp16(int total, const uint8_t *src, fp16 *dst)
{
    int threads = 256;
    int blocks = (total + threads * 8 - 1) / (threads * 8);
    kernel_cast_u8_to_fp16<<<blocks, threads>>>(total, src, cast(dst));
}

void launch_cast_u8_to_fp16_stream(int total, const uint8_t *src, fp16 *dst, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads * 8 - 1) / (threads * 8);
    kernel_cast_u8_to_fp16<<<blocks, threads, 0, stream>>>(total, src, cast(dst));
}

