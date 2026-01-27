#include "transpose.cuh"

const int REPEAT_TIME = 1;
const int BLOCK_X = 16, BLOCK_Y = 16;

template <typename T>
__global__ void transpose2d_naive_kernel(const T* input, T* output, const int M,
                                         const int N) {
  // global x and y
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_x >= N || tid_y >= M) return;
  T num = input[tid_y * N + tid_x];
  output[tid_x * M + tid_y] = num;
}

template <>
__global__ void transpose2d_vector_kernel(const float* input, float* output,
                                          const int M, const int N) {
  // global x and y
  int tid_x = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  int tid_y = 4 * (blockIdx.y * blockDim.y + threadIdx.y);

  if (N < tid_x || M < tid_y) return;
  // 每一个thread读取一个vector
  float4 src[4], dst[4];
  src[0] = reinterpret_cast<float4*>(
      const_cast<float*>(&input[(tid_y + 0) * N + tid_x]))[0];
  src[1] = reinterpret_cast<float4*>(
      const_cast<float*>(&input[(tid_y + 1) * N + tid_x]))[0];
  src[2] = reinterpret_cast<float4*>(
      const_cast<float*>(&input[(tid_y + 2) * N + tid_x]))[0];
  src[3] = reinterpret_cast<float4*>(
      const_cast<float*>(&input[(tid_y + 3) * N + tid_x]))[0];

  // 2d transpose 4x4
  dst[0].x = src[0].x;
  dst[0].y = src[1].x;
  dst[0].z = src[2].x;
  dst[0].w = src[3].x;

  dst[1].x = src[0].y;
  dst[1].y = src[1].y;
  dst[1].z = src[2].y;
  dst[1].w = src[3].y;

  dst[2].x = src[0].z;
  dst[2].y = src[1].z;
  dst[2].z = src[2].z;
  dst[2].w = src[3].z;

  dst[3].x = src[0].w;
  dst[3].y = src[1].w;
  dst[3].z = src[2].w;
  dst[3].w = src[3].w;

  // write
  reinterpret_cast<float4*>(&output[(tid_x + 0) * M + tid_y])[0] = dst[0];
  reinterpret_cast<float4*>(&output[(tid_x + 1) * M + tid_y])[0] = dst[1];
  reinterpret_cast<float4*>(&output[(tid_x + 2) * M + tid_y])[0] = dst[2];
  reinterpret_cast<float4*>(&output[(tid_x + 3) * M + tid_y])[0] = dst[3];
}

torch::Tensor transpose2d_naive(torch::Tensor X) {
  std::cout << "transpose2d_naive" << std::endl;
  const int M = X.size(0);
  const int N = X.size(1);
  std::cout << "M:" << M << " N:" << N << std::endl;

  auto Y = torch::zeros({N, M}, torch::kCUDA);
  dim3 block_dim(BLOCK_X, BLOCK_Y);
  dim3 grid_dim((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < REPEAT_TIME; ++i)
    transpose2d_naive_kernel<float><<<grid_dim, block_dim>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), M, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "transpose2d_naive Kernel execution time: "
            << milliseconds / REPEAT_TIME << " ms" << std::endl;

  return Y;
}

torch::Tensor transpose2d_vector(torch::Tensor X) {
  const int M = X.size(0);
  const int N = X.size(1);

  auto Y = torch::zeros({N, M}, torch::kCUDA);
  dim3 block_dim(BLOCK_X / 4, BLOCK_Y / 4);
  dim3 grid_dim((N + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < REPEAT_TIME; ++i)
    transpose2d_vector_kernel<float><<<grid_dim, block_dim>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), M, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "transpose2d_vector Kernel execution time: "
            << milliseconds / REPEAT_TIME << " ms" << std::endl;

  return Y;
}
