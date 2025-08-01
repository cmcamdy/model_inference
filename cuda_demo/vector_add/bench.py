import math
import os
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# https://zhenhuaw.me/blog/2019/gemm-optimization.html#%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E9%87%8F%E5%8C%96%E4%B8%AD%E7%9A%84%E7%9F%A9%E9%98%B5%E4%B9%98%E4%BC%98%E5%8C%96
# A800 1024*1024

# naive_gemm_kernel                   8.40042 ms
# naive_gemm_kernel(vector load)      2.35869 ms
# shared_gemm_kernel(TILE_SIZE 16)    0.51081 ms
# shared_gemm_kernel(TILE_SIZE 32)    0.47708 ms

# Load the CUDA kernel as a python module
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # 根据你的 GPU 架构调整
custom_add = load(name='custom_add', sources=['main.cpp', 'vector_add.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
seq_len = 32
head_embd = 32

seq_len = 1024
head_embd = 1024

q = torch.randn(seq_len).cuda()
k = torch.randn(seq_len).cuda()


print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def vector_add(A, B):
    return A+B

# with torch.autograd.profiler.profile(use_device="cuda") as prof:
#     manual_result = gemm(q, k)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
manual_result = vector_add(q, k)

print('=== profiling minimal flash attention === ')


with torch.autograd.profiler.profile(use_device="cuda") as prof:
    minimal_result = custom_add.vector_add(q, k)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result.float(), manual_result, rtol=1e-02, atol=1e-02))
# print(minimal_result)
# print(manual_result)