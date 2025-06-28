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
__global__ void smem_gemm_blocktiling_2d_vector_avoid_bank_conflict_kernel(
    const float* A, const float* B, float* C, const int M, const int N,
    const int K) {
  // Shared memory allocation
  // threadIdx.x一个处理TM个数据
  __shared__ float shared_A[TILE_SIZE_Y][TILE_SIZE_X];
  // __shared__ float shared_B[TILE_SIZE_Y][TILE_SIZE_X + 1];
  __shared__ float shared_B[TILE_SIZE_Y][TILE_SIZE_X];

  int ldx = threadIdx.x;
  // Calculate row and column index
  int row = TILE_SIZE_Y * blockIdx.y + TM * threadIdx.y;
  int col = TILE_SIZE_X * blockIdx.x + TN * threadIdx.x;

  int t_row = threadIdx.y * TM;
  int t_col = threadIdx.x * TN;
  int shuffle_bias;

  float value[TM * TN] = {0.0};
  float tmp_a[TM] = {0.0};
  float tmp_b[TN] = {0.0};

  for (int t = 0; t < (K + TILE_SIZE_X - 1) / TILE_SIZE_X; ++t) {
    // Load data into shared memory
    if (row < M && t * TILE_SIZE_X + t_col < K) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
          reinterpret_cast<float4*>(&shared_A[t_row + m][t_col + n])[0] =
              reinterpret_cast<float4*>(const_cast<float*>(
                  &A[(row + m) * K + t * TILE_SIZE_X + t_col + n]))[0];
        }
      }
    } else
      shared_A[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && t * TILE_SIZE_Y + threadIdx.y < K) {
      float4 tmp;
#pragma unroll
      for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
          //   reinterpret_cast<float4*>(&shared_B[t_row + m][t_col + n])[0] =
          //       reinterpret_cast<float4*>(const_cast<float*>(
          //           &B[(t * TILE_SIZE_Y + t_row + m) * N + col + n]))[0];
          shuffle_bias = (ldx * 4 + n) % TN;
          tmp = reinterpret_cast<float4*>(const_cast<float*>(
              &B[(t * TILE_SIZE_Y + t_row + m) * N + col + n]))[0];
          shared_B[t_row + m][t_col + shuffle_bias + 0] = tmp.x;
          shared_B[t_row + m][t_col + shuffle_bias + 1] = tmp.y;
          shared_B[t_row + m][t_col + shuffle_bias + 2] = tmp.z;
          shared_B[t_row + m][t_col + shuffle_bias + 3] = tmp.w;
        }
      }
    } else
      shared_B[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();
    // Perform computation
    // 从目前的样子看起来，可以对B作向量化，但是对A，可能改动会大点1115
    for (int i = 0; i < TILE_SIZE_X; i++) {
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        shuffle_bias = (ldx * 4 + n) % TN;
        tmp_b[n + 0] = shared_B[i][t_col + shuffle_bias + 0];
        tmp_b[n + 1] = shared_B[i][t_col + shuffle_bias + 1];
        tmp_b[n + 2] = shared_B[i][t_col + shuffle_bias + 2];
        tmp_b[n + 3] = shared_B[i][t_col + shuffle_bias + 3];
      }
#pragma unroll
      for (int m = 0; m < TM; m++) {
        tmp_a[m] = shared_A[t_row + m][i];
      }

#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          value[m * TN + n] += tmp_a[m] * tmp_b[n];
        }
      }
    }
    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        reinterpret_cast<float4*>(&C[(row + m) * N + col + n])[0] =
            reinterpret_cast<float4*>(&value[m * TN + n])[0];
      }
    }
  }
}


torch::Tensor smem_gemm_blocktiling_2d_vector_avoid_bank_conflict(
    torch::Tensor A, torch::Tensor B) {
  const int block_size_x = TILE_SIZE_X;
  const int block_size_y = TILE_SIZE_Y;
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  auto C = torch::zeros({M, N}, torch::kCUDA);  // 确保在 CUDA 设备上分配内存
  dim3 grid_dim((M + block_size_x - 1) / block_size_x,
                (N + block_size_y - 1) / block_size_y);
  // 按照x y z这个顺序排的
  dim3 block_dim(block_size_x / TN, block_size_y / TM);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < REPEAT_TIME; ++i)
    smem_gemm_blocktiling_2d_vector_avoid_bank_conflict_kernel<float>
        <<<grid_dim, block_dim>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                  C.data_ptr<float>(), M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "smem_gemm_blocktiling_2d_vector_avoid_bank_conflict Kernel "
               "execution time: "
            << milliseconds / REPEAT_TIME << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
}
