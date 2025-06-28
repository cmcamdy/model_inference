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
  for (int i = 0; i < REPEAT_TIME; ++i)
    naive_gemm_kernel<float><<<grid_dim, block_dim, sram_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "naive_gemm Kernel execution time: "
            << milliseconds / REPEAT_TIME << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return C;
}
