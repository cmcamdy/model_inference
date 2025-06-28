#include <cuda_runtime.h>

#include "gemm.cuh"

const int TN = 4;
const int TM = 4;
const int TILE_SIZE = 32;
const int TILE_SIZE_X = 64;
const int TILE_SIZE_Y = 64;
// 用于测速
const int REPEAT_TIME = 1000;

template <>
__global__ void smem_gemm_blocktiling_1d_kernel(const float* A, const float* B,
                                                float* C, const int M,
                                                const int N, const int K) {
  // Shared memory allocation
  // threadIdx.x一个处理TM个数据
  __shared__ float shared_A[TILE_SIZE_Y][TILE_SIZE_X];
  __shared__ float shared_B[TILE_SIZE_Y][TILE_SIZE_X];

  // Calculate row and column index
  int row = TILE_SIZE_Y * blockIdx.y + TM * threadIdx.y;
  int col = TILE_SIZE_X * blockIdx.x + threadIdx.x;

  int t_row = threadIdx.y * TM;
  float value[TM] = {0.0};

  for (int t = 0; t < (K + TILE_SIZE_X - 1) / TILE_SIZE_X; ++t) {
    // Load data into shared memory
    if (row < M && t * TILE_SIZE_X + threadIdx.x < K) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
        shared_A[t_row + m][threadIdx.x] =
            A[(row + m) * K + t * TILE_SIZE_X + threadIdx.x];
      }
    } else
      shared_A[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && t * TILE_SIZE_Y + threadIdx.y < K) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
        shared_B[t_row + m][threadIdx.x] =
            B[(t * TILE_SIZE_X + t_row + m) * N + col];
      }
    } else
      shared_B[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Perform computation
    for (int i = 0; i < TILE_SIZE_X; ++i) {
      float tmp_b = shared_B[i][threadIdx.x];
#pragma unroll
      for (int m = 0; m < TM; m++) {
        value[m] += shared_A[t_row + m][i] * tmp_b;
      }
    }
    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
#pragma unroll
    for (int m = 0; m < TM; m++) {
      C[(row + m) * N + col] = value[m];
    }
  }
}


torch::Tensor smem_gemm_blocktiling_1d(torch::Tensor A, torch::Tensor B) {
  // const int block_size = 32;
  const int block_size_x = TILE_SIZE_X;
  const int block_size_y = TILE_SIZE_Y;
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);
  auto C = torch::zeros({M, N}, torch::kCUDA);  // 确保在 CUDA 设备上分配内存
  dim3 grid_dim((M + block_size_x - 1) / block_size_x,
                (N + block_size_y) / block_size_y);
  // 按照x y z这个顺序排的
  dim3 block_dim(block_size_x, block_size_y / TM);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < REPEAT_TIME; ++i)
    smem_gemm_blocktiling_1d_kernel<float><<<grid_dim, block_dim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "smem_gemm_blocktiling_1d Kernel execution time: "
            << milliseconds / REPEAT_TIME << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
}