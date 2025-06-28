#include <cuda_runtime.h>

#include "gemm.cuh"

const int TN = 4;
const int TM = 4;
const int TILE_SIZE = 32;
const int TILE_SIZE_X = 64;
const int TILE_SIZE_Y = 64;
// 用于测速
const int REPEAT_TIME = 1000;

// Define tile size
template <>
__global__ void smem_gemm_kernel(const float* A, const float* B, float* C,
                                 const int M, const int N, const int K) {
  // Shared memory allocation
  __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

  // Calculate row and column index
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float value = 0;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load data into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < K)
      shared_A[threadIdx.y][threadIdx.x] =
          A[row * K + t * TILE_SIZE + threadIdx.x];
    else
      shared_A[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && t * TILE_SIZE + threadIdx.y < K)
      shared_B[threadIdx.y][threadIdx.x] =
          B[(t * TILE_SIZE + threadIdx.y) * N + col];
    else
      shared_B[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Perform computation
    for (int i = 0; i < TILE_SIZE; ++i) {
      value += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
    }

    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
    C[row * N + col] = value;
  }
}

torch::Tensor smem_gemm(torch::Tensor A, torch::Tensor B) {
  // const int block_size = 32;
  const int block_size = TILE_SIZE;
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  auto C = torch::zeros({M, N}, torch::kCUDA);  // 确保在 CUDA 设备上分配内存
  dim3 grid_dim((M + block_size - 1) / block_size,
                (N + block_size - 1) / block_size);
  dim3 block_dim(block_size, block_size);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < REPEAT_TIME; ++i)

    smem_gemm_kernel<float><<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "smem_gemm Kernel execution time: " << milliseconds / REPEAT_TIME
            << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
}
