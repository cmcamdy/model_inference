#include "softmax.cuh"

// const int REPEAT_TIME = 1000;
const int BLOCK_X = 256;

__device__ static inline float warp_reduce_sum(float sum) {
  sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  sum += __shfl_down_sync(0xffffffff, sum, 8);   // 0-8, 1-9, 2-10, etc.
  sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, 2-6, etc.
  sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, 4-6, 5-7, etc.
  sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1, 2-3, 4-5, etc.
  return sum;
}

__device__ static inline float warp_reduce_max(float sum) {
  sum = max(sum,
            __shfl_down_sync(0xffffffff, sum, 16));  // 0-16, 1-17, 2-18, etc.
  sum = max(sum, __shfl_down_sync(0xffffffff, sum, 8));  // 0-8, 1-9, 2-10, etc.
  sum = max(sum, __shfl_down_sync(0xffffffff, sum, 4));  // 0-4, 1-5, 2-6, etc.
  sum = max(sum,
            __shfl_down_sync(0xffffffff, sum, 2));  // 0-2, 1-3, 4-6, 5-7, etc.
  sum = max(sum, __shfl_down_sync(0xffffffff, sum, 1));  // 0-1, 2-3, 4-5, etc.
  return sum;
}

__device__ static inline void block_reduce_max(float* smem) {
  int tid = threadIdx.x;
  if (BLOCK_X > 256) {
    if (tid < 128) {
      smem[tid] = max(smem[tid], smem[tid + 128]);
      __syncthreads();
    }
  }

  if (BLOCK_X > 128) {
    if (tid < 64) {
      smem[tid] = max(smem[tid], smem[tid + 64]);
      __syncthreads();
    }
  }

  if (BLOCK_X > 64) {
    if (tid < 32) {
      smem[tid] = max(smem[tid], smem[tid + 32]);
      __syncthreads();
    }
  }
}

__device__ static inline void block_reduce_sum(float* smem) {
  int tid = threadIdx.x;
  if (BLOCK_X > 256) {
    if (tid < 128) {
      smem[tid] += smem[tid + 128];
      __syncthreads();
    }
  }

  if (BLOCK_X > 128) {
    if (tid < 64) {
      smem[tid] += smem[tid + 64];
      __syncthreads();
    }
  }

  if (BLOCK_X > 64) {
    if (tid < 32) {
      smem[tid] += smem[tid + 32];
      __syncthreads();
    }
  }
}

template <>
__global__ void softmax_2d_naive_kernel(const float* input, float* output,
                                        const int M, const int N) {
  // global x, 只需要1D
  int tid = threadIdx.x;
  const float* input_now = input + blockIdx.x * N;
  float* output_now = output + blockIdx.x * N;
  __shared__ float r_max[BLOCK_X];
  __shared__ float r_sum[BLOCK_X];
  float global_max = 0;
  float global_sum;
  // 1.reduce max
#pragma unroll
  for (int i = tid; i < N; i += blockDim.x) {
    float tmp = input_now[i];
    r_max[tid] = r_max[tid] > tmp ? r_max[tid] : tmp;
  }

  block_reduce_max(r_max);
  if (tid < 32) {
    r_max[0] = warp_reduce_max(r_max[tid]);
  }
  __syncthreads();
  global_max = r_max[0];

  // 2.exp + reduce sum
  for (int i = tid; i < N; i += blockDim.x) {
    float tmp = input_now[i];
    r_sum[tid] += __expf(tmp - global_max);
  }
  block_reduce_sum(r_sum);

  if (tid < 32) {
    r_sum[0] = warp_reduce_sum(r_sum[tid]);
  }
  __syncthreads();
  global_sum = r_sum[0];
  // 3.div & 4.save
  for (int i = tid; i < N; i += blockDim.x) {
    float tmp = input_now[i];
    output_now[i] = exp(tmp - global_max) / global_sum;
  }
}

torch::Tensor softmax_2d_naive(torch::Tensor X, const int REPEAT_TIME=1) {
  std::cout << "softmax_2d_naive" << std::endl;
  const int M = X.size(0);
  const int N = X.size(1);
  std::cout << "M:" << M << " N:" << N << std::endl;

  auto Y = torch::zeros({M, N}, torch::kCUDA);
  dim3 block_dim(BLOCK_X);
  dim3 grid_dim(M);
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < REPEAT_TIME; ++i)
    softmax_2d_naive_kernel<float><<<grid_dim, block_dim>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), M, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "softmax_2d_naive Kernel execution time: "
            << milliseconds / REPEAT_TIME << " ms" << std::endl;

  return Y;
}
