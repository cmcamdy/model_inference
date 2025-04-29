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

// 对TM、TN做till
template <typename T>
__global__ void smem_gemm_blocktiling_2d_kernel(const T* A, const T* B, T* C,
                                                const int M, const int N,
                                                const int K);
