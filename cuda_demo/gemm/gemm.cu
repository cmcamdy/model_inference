#include <cuda_runtime.h>

#include "gemm.cuh"

const int TN = 4;
const int TM = 4;
const int TILE_SIZE = 32;
const int TILE_SIZE_X = 64;
const int TILE_SIZE_Y = 64;

template <>
__global__ void naive_gemm_kernel(const float* A, const float* B, float* C,
                                  const int M, const int N, const int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < M && idy < N) {  // 确保索引在矩阵范围内
    float sum = 0.0;
    // // 1024*1024 8ms
    // 一个thread负责K个元素
    // for(int k=0; k<K; k+=4){
    //     // float4 tmp_a =
    //     reinterpret_cast<float4*>(const_cast<float*>(A))[idx*K / 4 + k];
    //     // printf("check A: %d vs %d\n", tmp_a.x, A[idx*K + k]);
    //     // sum += tmp_a.x * B[(k+0)*N + idy];
    //     // sum += tmp_a.y * B[(k+1)*N + idy];
    //     // sum += tmp_a.z * B[(k+2)*N + idy];
    //     // sum += tmp_a.w * B[(k+3)*N + idy];
    //     sum += A[idx*K + k+0] * B[(k+0)*N + idy];
    //     sum += A[idx*K + k+1] * B[(k+1)*N + idy];
    //     sum += A[idx*K + k+2] * B[(k+2)*N + idy];
    //     sum += A[idx*K + k+3] * B[(k+3)*N + idy];
    // }

    // 1024*1024 2.4ms
    // #pragma unroll
    for (int k = 0; k < K / 4; k++) {
      float4 tmp_a =
          reinterpret_cast<float4*>(const_cast<float*>(A))[idx * K / 4 + k];
      // printf("check A: %d vs %d\n", tmp_a.x, A[idx*K + k]);
      sum += tmp_a.x * B[(k * 4 + 0) * N + idy];
      sum += tmp_a.y * B[(k * 4 + 1) * N + idy];
      sum += tmp_a.z * B[(k * 4 + 2) * N + idy];
      sum += tmp_a.w * B[(k * 4 + 3) * N + idy];
    }

    C[idx * N + idy] = sum;
  }
}

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

template <>
__global__ void smem_gemm_blocktiling_2d_kernel(const float* A, const float* B,
                                                float* C, const int M,
                                                const int N, const int K) {
  // Shared memory allocation
  // threadIdx.x一个处理TM个数据
  __shared__ float shared_A[TILE_SIZE_Y][TILE_SIZE_X];
  __shared__ float shared_B[TILE_SIZE_Y][TILE_SIZE_X];

  // Calculate row and column index
  int row = TILE_SIZE_Y * blockIdx.y + TM * threadIdx.y;
  int col = TILE_SIZE_X * blockIdx.x + TN * threadIdx.x;

  int t_row = threadIdx.y * TM;
  int t_col = threadIdx.x * TN;

  float value[TM * TN] = {0.0};
  // float tmp_a[TN] = {0.0};
  float tmp_a[TM] = {0.0};
  float tmp_b[TN] = {0.0};

  for (int t = 0; t < (K + TILE_SIZE_X - 1) / TILE_SIZE_X; ++t) {
    // Load data into shared memory
    if (row < M && t * TILE_SIZE_X + t_col < K) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
          shared_A[t_row + m][t_col + n] =
              A[(row + m) * K + t * TILE_SIZE_X + t_col + n];
        }
      }
    } else
      shared_A[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && t * TILE_SIZE_Y + threadIdx.y < K) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
          shared_B[t_row + m][t_col + n] =
              B[(t * TILE_SIZE_Y + t_row + m) * N + col + n];
        }
      }
    } else
      shared_B[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();
    // Perform computation
    for (int i = 0; i < TILE_SIZE_X; i++) {
#pragma unroll
      for (int n = 0; n < TN; n++) {
        tmp_b[n] = shared_B[i][t_col + n];
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
      for (int n = 0; n < TN; n++) {
        C[(row + m) * N + col + n] = value[m * TN + n];
      }
    }
  }
}

torch::Tensor naive_gemm(torch::Tensor A, torch::Tensor B) {
  const int block_size = 32;
  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1);

  auto C = torch::zeros({M, N}, torch::kCUDA);  // 确保在 CUDA 设备上分配内存
  dim3 grid_dim((M + block_size - 1) / block_size,
                (N + block_size - 1) / block_size);
  dim3 block_dim(block_size, block_size);
  const int sram_size = 0;

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  naive_gemm_kernel<float><<<grid_dim, block_dim, sram_size>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "naive_gemm Kernel execution time: " << milliseconds << " ms"
            << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
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

  smem_gemm_kernel<float><<<grid_dim, block_dim>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "smem_gemm Kernel execution time: " << milliseconds << " ms"
            << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
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

  smem_gemm_blocktiling_1d_kernel<float><<<grid_dim, block_dim>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "smem_gemm_blocktiling_1d Kernel execution time: "
            << milliseconds << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
}

torch::Tensor smem_gemm_blocktiling_2d(torch::Tensor A, torch::Tensor B) {
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

  smem_gemm_blocktiling_2d_kernel<float><<<grid_dim, block_dim>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "smem_gemm_blocktiling_2d Kernel execution time: "
            << milliseconds << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
}