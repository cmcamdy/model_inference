#include <ATen/Tensor.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/types.h>

#include <ATen/cuda/CUDATensorMethods.cuh>

__global__ void vector_add_kernel(const float* A, const float* B, float* C,
                                  const int M) {
  int idx = threadIdx.x;
  int stride = blockDim.x;
  for (int i = idx; i < M; i += stride) {
    C[i] = A[i] + B[i];
  }
}

torch::Tensor vector_add(torch::Tensor A, torch::Tensor B) {
  const int block_size = 256;
  const int M = A.size(0);

  auto C = torch::zeros({M}, torch::kCUDA);  // 确保在 CUDA 设备上分配内存

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  vector_add_kernel<<<1, 256>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                C.data_ptr<float>(), M);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
}