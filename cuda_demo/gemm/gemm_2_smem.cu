#include <cuda_runtime.h>

#include "gemm.cuh"

const int TILE_SIZE = 16;
const int TILE_SIZE_M = 32;
const int TILE_SIZE_N = 32;
const int TILE_SIZE_K = 16;
const int moveNumN = 5;
const int moveNumK = 3;

const int TM = TILE_SIZE_M / TILE_SIZE;
const int TN = TILE_SIZE_N / TILE_SIZE;

// 用于测速
const int REPEAT_TIME = 1;

// Define tile size
template <>
__global__ void smem_gemm_kernel(const float* A, const float* B, float* C,
                                 const int M, const int N, const int K) {
  // Shared memory allocation
  __shared__ float shared_A[TILE_SIZE_M][TILE_SIZE_K];
  __shared__ float shared_B[TILE_SIZE_K][TILE_SIZE_N];

  // block 内部的tid
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // C row and column index
  int row = blockIdx.y * TILE_SIZE_M + TM * threadIdx.y;
  int col = blockIdx.x * TILE_SIZE_N + TN * threadIdx.x;

  const float* A_base = A + blockIdx.y * TILE_SIZE_M * K;
  const float* B_base = B + blockIdx.x * TILE_SIZE_N;

  // float reg_A[TM][TILE_SIZE_K] = {};
  // float reg_B[TILE_SIZE_K][TN] = {};
  float value[TM][TN] = {0};
#pragma unroll
  for (int t = 0; t < (K + TILE_SIZE_K - 1) / TILE_SIZE_K; ++t) {
    // load data
    // 对于A来说，一次blockDim.x * blockDim.y / TILE_SIZE_K行
    // col = threadIdx.x % 8 >> (threadIdx.x & 7)
    // row = (threadIdx.y * (blockDim.x / TILE_SIZE_K) + threadIdx.x / 8)
#pragma unroll
    for (int step = tid; step < TILE_SIZE_M * TILE_SIZE_K;
         step += blockDim.x * blockDim.y) {
        int col = step % TILE_SIZE_K;
        int row = step / TILE_SIZE_K;
        shared_A[row][col] = A_base[t * TILE_SIZE_K + row * K + col];
      // int col = step % TILE_SIZE_K / 4;
      // int row = step / TILE_SIZE_K;
      // int col = step & (TILE_SIZE_K-1);
      // int row = step >> moveNumK;
      // reinterpret_cast<float4*>(&shared_A[row][col*4])[0] =
      //     reinterpret_cast<float4*>(
      //         const_cast<float*>(&A_base[t * TILE_SIZE_K + row * K + col*4]))[0];
    }

    // 对于B来说，一次blockDim.x * blockDim.y / TILE_SIZE_N行
    for (int step = tid; step < TILE_SIZE_K * TILE_SIZE_N;
         step += blockDim.x * blockDim.y) {
      int col = step % TILE_SIZE_N;
      int row = step / TILE_SIZE_N;
      // int col = step & (TILE_SIZE_N-1);
      // int row = step >> moveNumN;
      shared_B[row][col] = B_base[(t * TILE_SIZE_K) * N + row * N + col];
    }
    __syncthreads();

    // read to reg
    // 64*64 / 16*16 => 一个thread要搞16个数
    for (int k = 0; k < TILE_SIZE_K; k++) {
#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
          value[i][j] += shared_A[threadIdx.y * TM + i][k] *
                         shared_B[k][threadIdx.x * TN + j];
        }
      }
    }
    // __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
    for (int i = 0; i < TM; i++) {
      for (int j = 0; j < TN; j++) {
        C[(row + i) * N + col + j] = value[i][j];
      }
    }
  }
}

torch::Tensor smem_gemm(torch::Tensor A, torch::Tensor B) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  auto C = torch::zeros({M, N}, torch::kCUDA);  // 确保在 CUDA 设备上分配内存
  dim3 grid_dim((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);
  dim3 block_dim(TILE_SIZE, TILE_SIZE);

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
