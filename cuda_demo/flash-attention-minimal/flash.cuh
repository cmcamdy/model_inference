#include <ATen/Tensor.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <ATen/cuda/CUDATensorMethods.cuh>

template <typename T>
__global__ void flash_attention_forward_kernel(
    const T* Q, const T* K, const T* V, const int N, const int d, const int Tc,
    const int Tr, const int Bc, const int Br, const float softmax_scale,
    float* l, float* m, T* O);
