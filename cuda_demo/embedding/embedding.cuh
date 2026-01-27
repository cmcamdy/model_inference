#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#include <ATen/Tensor.h>
#include <ATen/cuda/CUDATensorMethods.cuh>


__global__ void embedding_naive_kernel(const int* input, float* weight, float* output, const int n, const int emb_size);