#ifndef UTILS_CUDA_UTILS_H_
#define UTILS_CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(condition)                                                                     \
  /* Code block avoids redefinition of cudaError_t error */                                       \
  do                                                                                              \
  {                                                                                               \
    cudaError_t error = condition;                                                                \
    if (error != cudaSuccess)                                                                     \
    {                                                                                             \
      std::cout << cudaGetErrorString(error) << " " << __FILE__ << "  " << __LINE__ << std::endl; \
      std::exit(-1);                                                                              \
    }                                                                                             \
  } while (0)

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

#define CUDA_1D_KERNEL_LOOP_EX(i, n, DType) \
  for (DType i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

// TODO: check if the choice of this arugment affect the performance.
#define CUDA_NUM_THREADS 128
// TODO: check the choice of this argument affect the performance.
#define CUDA_MAXIMUM_NUM_BLOCKS 4096

// Check if the kernel launch correctly.
#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define CUDA_GET_BLOCKS(N)                                                            \
  static_cast<int>(std::max(std::min(static_cast<size_t>(DIVUP(N, CUDA_NUM_THREADS)), \
                                     static_cast<size_t>(CUDA_MAXIMUM_NUM_BLOCKS)),   \
                            static_cast<size_t>(1)))

#define CUDA_GET_BLOCKS_EX(N, M)                                                                \
  static_cast<int>(std::max(                                                                    \
      std::min(static_cast<size_t>(DIVUP(N, M)), static_cast<size_t>(CUDA_MAXIMUM_NUM_BLOCKS)), \
      static_cast<size_t>(1)))

template <size_t device>
class CUDADeviceProperties
{
public:
  static const cudaDeviceProp &getInstance()
  {
    static CUDADeviceProperties instance;
    return instance.deviceProp;
  }

private:
  cudaDeviceProp deviceProp;
  CUDADeviceProperties() { cudaGetDeviceProperties(&deviceProp, device); }

public:
  CUDADeviceProperties(const CUDADeviceProperties &) = delete;
  CUDADeviceProperties &operator=(const CUDADeviceProperties &) = delete;
};

#endif // UTILS_CUDA_UTILS_H_
