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
