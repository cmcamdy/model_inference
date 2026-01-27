#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/types.h>

#include <algorithm>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val, float* smem) {
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warpid = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) {
    smem[warpid] = val;
  }
  __syncthreads();
  val = (lane < NUM_WARPS) ? smem[lane] : 0;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  return val;
}

// RMS Norm: x: NxK(K=256<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template <const int NUM_THREADS = 256>
__global__ void rms_norm_naive_kernel(float* x, float* y, float g, int N,
                                      int K) {
  int tid = threadIdx.x;
  float val = 0;
  float x_1 = 0;
  __shared__ float r_rms[1];
  // 相当于是一个block搞定一行，一行是K个
  float* x_now = x + blockIdx.x * K;
  float* y_now = y + blockIdx.x * K;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  static __shared__ float smem[NUM_WARPS];

  // 1.rms
  x_1 = x_now[tid];
  val = block_reduce_sum_f32<NUM_THREADS>(x_1 * x_1, smem);

  if (threadIdx.x == 0) {
    // warp0,line0
    r_rms[0] = rsqrtf(val / K);
  }
  __syncthreads();
  // reuse val, as r_rms
  val = r_rms[0];

  // 2. norm
  // 3. y' * g
  y_now[tid] = g * (x_1 * val);
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_vector_kernel(float* x, float* y, float g, int N,
                                       int K) {
  int tid = 4 * threadIdx.x;
  float val = 0;
  float4 x_1;
  float4 y_1;
  __shared__ float r_rms[1];
  // 相当于是一个block搞定一行，一行是K个
  float* x_now = x + blockIdx.x * K;
  float* y_now = y + blockIdx.x * K;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  static __shared__ float smem[NUM_WARPS];

  // 1.rms
  x_1 = FLOAT4(x_now[tid]);

  val += x_1.x * x_1.x;
  val += x_1.y * x_1.y;
  val += x_1.z * x_1.z;
  val += x_1.w * x_1.w;

  // if(tid == 0) printf("ch -- debug val:%f\n",val);
  val = block_reduce_sum_f32<NUM_THREADS>(val, smem);
  // if(tid == 0) printf("ch -- debug val:%f\n",val);

  if (threadIdx.x == 0) {
    // warp0,line0
    r_rms[0] = rsqrtf(val / K);
  }
  __syncthreads();
  // reuse val, as r_rms
  val = r_rms[0];
  // if(tid == 0) printf("ch -- debug val:%f\n",val);

  // 2. norm
  // 3. y' * g
  y_1.x = x_1.x * val * g;
  y_1.y = x_1.y * val * g;
  y_1.z = x_1.z * val * g;
  y_1.w = x_1.w * val * g;
  FLOAT4(y_now[tid]) = y_1;
}

const int NUM_X1 = 2;
template <const int NUM_THREADS = 256>
__global__ void rms_norm_vector_v2_kernel(float* x, float* y, float g, int N,
                                       int K) {
  // assert NUM_X1 > K / NUM_THREADS
  int step = K / NUM_THREADS / 4;
  int tid = 4 * threadIdx.x;
  float val = 0;
  float4 x_1[NUM_X1];
  float4 y_1;
  __shared__ float r_rms[1];
  // 相当于是一个block搞定一行，一行是K个
  float* x_now = x + blockIdx.x * K;
  float* y_now = y + blockIdx.x * K;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  static __shared__ float smem[NUM_WARPS];

  // 1.rms
  int offset = tid;
#pragma unroll
  for (size_t i = 0; i < step; i++) {
    x_1[i] = FLOAT4(x_now[offset]);
    val += x_1[i].x * x_1[i].x;
    val += x_1[i].y * x_1[i].y;
    val += x_1[i].z * x_1[i].z;
    val += x_1[i].w * x_1[i].w;
    offset += blockDim.x * 4;
  }

  val = block_reduce_sum_f32<NUM_THREADS>(val, smem);
  if (threadIdx.x == 0) {
    // warp0,line0
    r_rms[0] = rsqrtf(val / K);
  }
  __syncthreads();
  // reuse val, as r_rms
  val = r_rms[0];
  // 2. norm
  // 3. y' * g
  offset = tid;
#pragma unroll
  for (size_t i = 0; i < step; i++) {
    y_1.x = x_1[i].x * val * g;
    y_1.y = x_1[i].y * val * g;
    y_1.z = x_1[i].z * val * g;
    y_1.w = x_1[i].w * val * g;
    FLOAT4(y_now[offset]) = y_1;
    offset += blockDim.x * 4;
  }
}




#define LANUCH_RMS_NORM_F32_KERNEL(K)                           \
  rms_norm_naive_kernel<(K)>                                    \
      <<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()), \
                        reinterpret_cast<float*>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F32_KERNEL(N, K)                             \
  dim3 block((K));                                                     \
  dim3 grid((N));                                                      \
                                                                       \
  switch ((K)) {                                                       \
    case 64:                                                           \
      LANUCH_RMS_NORM_F32_KERNEL(64)                                   \
      break;                                                           \
    case 128:                                                          \
      LANUCH_RMS_NORM_F32_KERNEL(128)                                  \
      break;                                                           \
    case 256:                                                          \
      LANUCH_RMS_NORM_F32_KERNEL(256)                                  \
      break;                                                           \
    case 512:                                                          \
      LANUCH_RMS_NORM_F32_KERNEL(512)                                  \
      break;                                                           \
    case 1024:                                                         \
      LANUCH_RMS_NORM_F32_KERNEL(1024)                                 \
      break;                                                           \
    default:                                                           \
      throw std::runtime_error("only support K: 64/128/256/512/1024"); \
      break;                                                           \
  }


#define LANUCH_RMS_NORM_F32X4_KERNEL(K)                         \
  rms_norm_vector_kernel<(K / 4)>                               \
      <<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()), \
                        reinterpret_cast<float*>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F32X4_KERNEL(N, K)                           \
  dim3 block((K / 4));                                                 \
  dim3 grid((N));                                                      \
                                                                       \
  switch ((K)) {                                                       \
    case 64:                                                           \
      LANUCH_RMS_NORM_F32X4_KERNEL(64)                                 \
      break;                                                           \
    case 128:                                                          \
      LANUCH_RMS_NORM_F32X4_KERNEL(128)                                \
      break;                                                           \
    case 256:                                                          \
      LANUCH_RMS_NORM_F32X4_KERNEL(256)                                \
      break;                                                           \
    case 512:                                                          \
      LANUCH_RMS_NORM_F32X4_KERNEL(512)                                \
      break;                                                           \
    case 1024:                                                         \
      LANUCH_RMS_NORM_F32X4_KERNEL(1024)                               \
      break;                                                           \
    case 2048:                                                         \
      LANUCH_RMS_NORM_F32X4_KERNEL(2048)                               \
      break;                                                           \
    case 4096:                                                         \
      LANUCH_RMS_NORM_F32X4_KERNEL(4096)                               \
      break;                                                           \
    default:                                                           \
      throw std::runtime_error("only support K: 64/128/256/512/1024"); \
      break;                                                           \
  }

  #define LANUCH_RMS_NORM_F32X4V2_KERNEL(num_thread)                      \
  rms_norm_vector_v2_kernel<(num_thread)>                               \
      <<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()),           \
                        reinterpret_cast<float*>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F32X4V2_KERNEL(N, K)                           \
  dim3 grid((N));                                                      \
  dim3 block((min(1024, K/4)));                                          \
                                                                       \
  switch ((K)) {                                                       \
    case 64:                                                           \
      LANUCH_RMS_NORM_F32X4V2_KERNEL(16)                                 \
      break;                                                           \
    case 128:                                                          \
      LANUCH_RMS_NORM_F32X4V2_KERNEL(32)                                \
      break;                                                           \
    case 256:                                                          \
      LANUCH_RMS_NORM_F32X4V2_KERNEL(64)                                \
      break;                                                           \
    case 512:                                                          \
      LANUCH_RMS_NORM_F32X4V2_KERNEL(128)                                \
      break;                                                           \
    case 1024:                                                         \
      LANUCH_RMS_NORM_F32X4V2_KERNEL(256)                               \
      break;                                                           \
    case 2048:                                                         \
      LANUCH_RMS_NORM_F32X4V2_KERNEL(512)                               \
      break;                                                           \
    case 4096:                                                         \
      LANUCH_RMS_NORM_F32X4V2_KERNEL(1024)                               \
      break;                                                           \
    case 8192:                                                         \
      LANUCH_RMS_NORM_F32X4V2_KERNEL(1024)                               \
      break;                                                           \
    default:                                                           \
      throw std::runtime_error("only support K: 64/128/256/512/1024"); \
      break;                                                           \
  }

void rms_norm_f32(torch::Tensor x, torch::Tensor y, float g) {
  const int N = x.size(0);
  const int K = x.size(1);
  dim3 block((K));
  dim3 grid((N));
  rms_norm_naive_kernel<4096>
      <<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()),
                        reinterpret_cast<float*>(y.data_ptr()), g, N, K);
  // std::cout << "ch -- debug K:" << K << std::endl;
  // DISPATCH_RMS_NORM_F32_KERNEL(N, K)
}

void rms_norm_f32x4(torch::Tensor x, torch::Tensor y, float g) {
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F32X4_KERNEL(N, K)
}

void rms_norm_f32x4v2(torch::Tensor x, torch::Tensor y, float g) {
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_RMS_NORM_F32X4V2_KERNEL(N, K)
}
