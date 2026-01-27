#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#include <ATen/Tensor.h>
#include <ATen/cuda/CUDATensorMethods.cuh>

template <typename T>
__global__ void transpose2d_naive_kernel(const T* input, T* output, const int M, const int N);


template <typename T>
__global__ void transpose2d_vector_kernel(const T* input, T* output, const int M, const int N);
