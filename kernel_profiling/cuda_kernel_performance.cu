#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <ctime>
#include <cstdlib>

// 定义常量
const int TN = 4;
const int TM = 4;
const int TILE_SIZE_X = 64;
const int TILE_SIZE_Y = 64;

// 假设内核函数已经定义
__global__ void
smem_gemm_blocktiling_2d_vector_avoid_bank_conflict_double_buffer_kernel(
    const float* A, const float* B, float* C, const int M, const int N,
    const int K) {
  // Shared memory allocation
  // threadIdx.x一个处理TM个数据
  __shared__ float shared_A[TILE_SIZE_Y][TILE_SIZE_X];
  __shared__ float shared_B[TILE_SIZE_Y][TILE_SIZE_X];

  int ldx = threadIdx.x;
  // Calculate row and column index
  int row = TILE_SIZE_Y * blockIdx.y + TM * threadIdx.y;
  int col = TILE_SIZE_X * blockIdx.x + TN * threadIdx.x;

  int t_row = threadIdx.y * TM;
  int t_col = threadIdx.x * TN;

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
          tmp = reinterpret_cast<float4*>(const_cast<float*>(
              &B[(t * TILE_SIZE_Y + t_row + m) * N + col + n]))[0];
          shared_B[t_row + m][t_col + (ldx * 4 + n) % TN + 0] = tmp.x;
          shared_B[t_row + m][t_col + (ldx * 4 + n) % TN + 1] = tmp.y;
          shared_B[t_row + m][t_col + (ldx * 4 + n) % TN + 2] = tmp.z;
          shared_B[t_row + m][t_col + (ldx * 4 + n) % TN + 3] = tmp.w;
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
        tmp_b[n + 0] = shared_B[i][t_col + (ldx * 4 + n) % TN + 0];
        tmp_b[n + 1] = shared_B[i][t_col + (ldx * 4 + n) % TN + 1];
        tmp_b[n + 2] = shared_B[i][t_col + (ldx * 4 + n) % TN + 2];
        tmp_b[n + 3] = shared_B[i][t_col + (ldx * 4 + n) % TN + 3];
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
// 检查 CUDA 调用是否出错
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " - " << message << std::endl;
        assert(false);
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(0);  // 选择第一个设备，可根据需求修改
    } else {
        // 处理没有可用 GPU 设备的情况
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }
    // 矩阵尺寸
    int M = 1024;
    int N = 1024;
    int K = 1024;

    // 主机端矩阵内存分配
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];

    // 初始化矩阵
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 设备端矩阵内存分配
    float* d_A;
    float* d_B;
    float* d_C;
    checkCudaError(cudaMalloc((void**)&d_A, M * K * sizeof(float)), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, K * N * sizeof(float)), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, M * N * sizeof(float)), "cudaMalloc d_C");

    // 将数据从主机复制到设备
    checkCudaError(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_B to d_B");
    
    // 定义网格和线程块大小
    dim3 dimBlock(TILE_SIZE_X / TN, TILE_SIZE_Y / TM);
    dim3 dimGrid((N + TILE_SIZE_X - 1) / TILE_SIZE_X, (M + TILE_SIZE_Y - 1) / TILE_SIZE_Y);

    // 调用内核函数
    smem_gemm_blocktiling_2d_vector_avoid_bank_conflict_double_buffer_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    // 检查内核调用是否出错
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // 将结果从设备复制到主机
    checkCudaError(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy d_C to h_C");

    // 释放设备内存
    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_B), "cudaFree d_B");
    checkCudaError(cudaFree(d_C), "cudaFree d_C");

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}