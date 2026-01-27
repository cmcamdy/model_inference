#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <algorithm>
#include <vector>

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define BLOCK_SIZE 256
#define theta 10000.0f

__global__ void rope_f32_kernel(float* x, float* out, int seq_len, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float x1 = x[2 * tid];
  float x2 = x[2 * tid + 1];
  int token_pos = tid / N;
  int token_idx = tid % N;
  float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);
  out[2 * tid] = x1 * cos_v - x2 * sin_v;
  out[2 * tid + 1] = x1 * sin_v + x2 * cos_v;
}

void rope_f32(torch::Tensor x, torch::Tensor out) {
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size / 2);
  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                   seq_len, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_f32", torch::wrap_pybind_function(rope_f32), "rope_f32");
}