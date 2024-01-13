/**
 * @file vector_add_1d.cuh
 * @author cmcandy
 * @brief
 * @version 0.1
 * @date 2024-01-13
 *
 * @copyright Copyright (c) 2024
 *
 */

namespace kernel {
namespace reduce_sum {

namespace functor {

namespace cuda_kernels {

template <int blockSize>
__device__ void BlockSharedMemReduce(float* smem) {
  // 对v4
  // L45的for循环展开，以减去for循环中的加法指令，以及给编译器更多重排指令的空间
  if (blockSize >= 1024) {
    if (threadIdx.x < 512) {
      smem[threadIdx.x] += smem[threadIdx.x + 512];
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (threadIdx.x < 256) {
      smem[threadIdx.x] += smem[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (threadIdx.x < 128) {
      smem[threadIdx.x] += smem[threadIdx.x + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (threadIdx.x < 64) {
      smem[threadIdx.x] += smem[threadIdx.x + 64];
    }
    __syncthreads();
  }
  // the final warp
  if (threadIdx.x < 32) {
    volatile float* vshm = smem;
    if (blockDim.x >= 64) {
      vshm[threadIdx.x] += vshm[threadIdx.x + 32];
    }
    vshm[threadIdx.x] += vshm[threadIdx.x + 16];
    vshm[threadIdx.x] += vshm[threadIdx.x + 8];
    vshm[threadIdx.x] += vshm[threadIdx.x + 4];
    vshm[threadIdx.x] += vshm[threadIdx.x + 2];
    vshm[threadIdx.x] += vshm[threadIdx.x + 1];
  }
}

/**
 * @brief 
 * 
 * @tparam DType 
 * @tparam blockSize 
 * @param in_tensor_ptr 
 * @param output_ptr 
 * @param length 
 * @return __global__ 
 * TODO：reduce优化，比如试试warp level
 */
template <typename DType, int blockSize>
__global__ void reduce_sum_1d_kernel(const DType* in_tensor_ptr,
                                     DType* output_ptr, int length) {
  __shared__ float smem[blockSize];

  unsigned int idx = threadIdx.x;
  unsigned int gidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int total_thread_num = blockDim.x * gridDim.x;
  // s == 1
  float sum = 0.0f;
  for (int32_t i = gidx; i < length; i += total_thread_num) {
    sum += in_tensor_ptr[i];
  }
  smem[idx] = sum;
  __syncthreads();

  // compute: reduce in shared mem
  BlockSharedMemReduce<blockSize>(smem);

  if (idx == 0) output_ptr[blockIdx.x] = smem[0];
}
}  // namespace cuda_kernels

}  // namespace functor
}  // namespace reduce_sum
}  // namespace kernel
