#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/types.h>

#include <ATen/cuda/CUDATensorMethods.cuh>

template <typename T>
__global__ void naive_gemm_kernel(const T* A, const T* B, T* C, const int M,
                                  const int N, const int K);

template <typename T>
__global__ void smem_gemm_kernel(const T* A, const T* B, T* C, const int M,
                                 const int N, const int K);

// template <typename T>
// __global__ void smem_vector_gemm_kernel(const T* A, const T* B, T* C, const
// int M, const int N, const int K);

/**
    参考：https://github.com/PaddleJitLab/CUDATutorial/tree/develop/docs/07_optimize_matmul
    对TM做till
    该kernel 采用了一维 Thread Tile
   并行优化，即一个thread处理m维度的多行（TM行），这样做的好处是减少对B矩阵的的访问，从而减少总体的访存
*/
template <typename T>
__global__ void smem_gemm_blocktiling_1d_kernel(const T* A, const T* B, T* C,
                                                const int M, const int N,
                                                const int K);

/**
 * @brief 对TM、TN做till
 *
 * @tparam T
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @return __global__
 */
template <typename T>
__global__ void smem_gemm_blocktiling_2d_kernel(const T* A, const T* B, T* C,
                                                const int M, const int N,
                                                const int K);
/**
 * @brief 对TM、TN做till, 加上向量化
 * 所谓向量化，就是读取的时候不必每次只读取一个数据，而是一次读取多个，
 *  比如float按照float4的格式读取4个
 *  或者half有half2这种的
 * @tparam T
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @return __global__
 */
template <typename T>
__global__ void smem_gemm_blocktiling_2d_vector_kernel(const T* A, const T* B,
                                                       T* C, const int M,
                                                       const int N,
                                                       const int K);

/**
 * @brief 解决bank conflict的方法
 * https://www.zhihu.com/question/667972067/answer/3634692524
 *
 * @tparam T
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @return __global__
 */
template <typename T>
__global__ void smem_gemm_blocktiling_2d_vector_avoid_bank_conflict_kernel(
    const T* A, const T* B, T* C, const int M, const int N, const int K);

template <typename T>
__global__ void smem_gemm_blocktiling_2d_vector_avoid_bank_conflict_warp_tiling_kernel(
    const T* A, const T* B, T* C, const int M, const int N, const int K);

template <typename T>
__global__ void
smem_gemm_blocktiling_2d_vector_avoid_bank_conflict_double_buffer_kernel(
    const T* A, const T* B, T* C, const int M, const int N, const int K);
