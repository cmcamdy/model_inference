import math
import os
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# https://zhenhuaw.me/blog/2019/gemm-optimization.html#%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E9%87%8F%E5%8C%96%E4%B8%AD%E7%9A%84%E7%9F%A9%E9%98%B5%E4%B9%98%E4%BC%98%E5%8C%96
# A800 1024*1024
# naive_gemm_kernel                                 8.40042 ms
# naive_gemm_kernel(vector load)                    2.35869 ms
# shared_gemm_kernel(TILE_SIZE 16)                  0.51081 ms
# shared_gemm_kernel(TILE_SIZE 32)                  0.47708 ms
# smem_gemm_blocktiling_1d(TILE_SIZE_X 32, TILE_SIZE_Y 32, TM 4)         0.26752 ms
# smem_gemm_blocktiling_1d(TILE_SIZE_X 32, TILE_SIZE_Y 32, TM 8)         0.244352 ms
# smem_gemm_blocktiling_1d(TILE_SIZE_X 32, TILE_SIZE_Y 64, TM 16)        0.239968 ms
# smem_gemm_blocktiling_1d(TILE_SIZE_X 64, TILE_SIZE_Y 32, TM 8)         0.258272 ms
# smem_gemm_blocktiling_1d(TILE_SIZE_X 64, TILE_SIZE_Y 64, TM 16)        0.253568 ms

# smem_gemm_blocktiling_2d(TILE_SIZE_X 32, TILE_SIZE_Y 32, TN = 4, TM 4)        0.2384 ms
# smem_gemm_blocktiling_2d(TILE_SIZE_X 64, TILE_SIZE_Y 64, TN = 4, TM 4)        0.230208 ms
# smem_gemm_blocktiling_2d(TILE_SIZE_X 64, TILE_SIZE_Y 32, TN = 4, TM 8)        0.233952 ms
# smem_gemm_blocktiling_2d(TILE_SIZE_X 32, TILE_SIZE_Y 64, TN = 8, TM 4)        0.317792 ms


# 3060 1024*1024
# naive_gemm Kernel execution time: 5.82274 ms
# smem_gemm Kernel execution time: 2.24327 ms
# smem_gemm_blocktiling_1d Kernel execution time: 0.882974 ms
# smem_gemm_blocktiling_2d Kernel execution time: 0.467195 ms
# smem_gemm_blocktiling_2d_vector Kernel execution time: 0.422579 ms
# smem_gemm_blocktiling_2d_vector_avoid_bank_conflict Kernel execution time: 0.424194 ms

# 3060 2048*2048
# naive_gemm Kernel execution time: 45.3744 ms
# smem_gemm Kernel execution time: 17.2198 ms
# smem_gemm_blocktiling_1d Kernel execution time: 6.2827 ms
# smem_gemm_blocktiling_2d Kernel execution time: 3.23536 ms
# smem_gemm_blocktiling_2d_vector Kernel execution time: 3.02139 ms
# smem_gemm_blocktiling_2d_vector_avoid_bank_conflict Kernel execution time: 3.02069 ms

# Load the CUDA kernel as a python module
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # 根据你的 GPU 架构调整
sources = ['main.cpp', 'gemm_1_naive.cu', 'gemm_2_smem.cu', 'gemm_3_block_tiling_1d.cu', 'gemm_4_block_tiling_2d.cu', 'gemm_5_vector.cu', 'gemm_6_bank_conflict.cu']
custom_gemm = load(name='custom_gemm', sources=sources, extra_cuda_cflags=['-O2'])
torch.set_printoptions(threshold=float('inf'))

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16

seq_len = 8
head_embd = 8

# seq_len = 16
# head_embd = 16

# seq_len = 32
# head_embd = 32

seq_len = 64
head_embd = 64

# seq_len = 128
# head_embd = 128

# seq_len = 256
# head_embd = 256

# seq_len = 512
# head_embd = 512

seq_len = 1024
head_embd = 1024

seq_len = 2048
head_embd = 2048

q = torch.randn(seq_len, head_embd).cuda()
k = torch.randn(seq_len, head_embd).cuda()
v = torch.randn(seq_len, head_embd).cuda()

q = torch.ones(seq_len, head_embd).cuda()
k = torch.ones(seq_len, head_embd).cuda()

q = torch.arange(seq_len * head_embd, dtype=torch.float32).reshape(seq_len, head_embd).cuda()
k = torch.arange(seq_len * head_embd, dtype=torch.float32).reshape(seq_len, head_embd).cuda()
v = torch.arange(seq_len * head_embd, dtype=torch.float32).reshape(seq_len, head_embd).cuda()

# print(q,k)

print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def gemm(A, B):
    return torch.matmul(A, B)

# with torch.autograd.profiler.profile(use_device="cuda") as prof:
#     manual_result = gemm(q, k)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
manual_result = gemm(q, k)

print('=== profiling minimal flash attention === ')

minimal_result = custom_gemm.naive_gemm(q, k)
minimal_result = custom_gemm.smem_gemm(q, k)
minimal_result = custom_gemm.smem_gemm_blocktiling_1d(q, k)
minimal_result = custom_gemm.smem_gemm_blocktiling_2d(q, k)
minimal_result = custom_gemm.smem_gemm_blocktiling_2d_vector(q, k)
minimal_result = custom_gemm.smem_gemm_blocktiling_2d_vector_avoid_bank_conflict(q, k)

# # print('attn values sanity check:', torch.allclose(minimal_result.float(), manual_result, atol=1e-02))
# if (not torch.allclose(minimal_result.float(), manual_result, atol=1e-02)):
#     print('attn values sanity check:', False)
#     # print(minimal_result - manual_result)
#     print(torch.sum(minimal_result))

# print('attn values sanity check:', torch.allclose(minimal_result.float(), manual_result, rtol=1e-02, atol=1e-02))
# print(minimal_result - manual_result)
# print(torch.max(minimal_result - manual_result))
# print(torch.mean(minimal_result - manual_result))
# print(torch.sum(torch.abs(minimal_result - manual_result)>1))
# print(torch.abs(minimal_result - manual_result)>1)