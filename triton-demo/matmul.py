import triton 
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(A_ptr + (offs_m[:, None] * K + (offs_k[None, :] + k)))
        B = tl.load(B_ptr + ((offs_k[:, None] + k) * N + offs_n[None, :]))
        acc += tl.dot(A, B)

    tl.store(C_ptr + (offs_m[:, None] * N + offs_n[None, :]), acc)
# 创建一些示例数据
import torch
torch.manual_seed(42)
# 定义矩阵尺寸
M, N, K = 256, 256, 256
# M, N, K = 128, 128, 128
# M, N, K = 32, 32, 32
# M, N, K = 16, 16, 16
A = torch.randn((M, K), device='cuda')
B = torch.randn((K, N), device='cuda')
# B = torch.ones((K, N), device='cuda')
C = torch.zeros((M, N), device='cuda')

# 启动矩阵乘法 kernel
# matmul_kernel[(M//16, N//16)](A, B, C, M, N, K, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16)
matmul_kernel[(M//32, N//32)](A, B, C, M, N, K, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=16)

# print(C)
# print(torch.matmul(A, B))
# print(C - torch.matmul(A, B))
print((C - torch.matmul(A, B)) / torch.matmul(A, B))
# 验证结果
assert torch.allclose(C, torch.matmul(A, B), rtol=1e-3, atol=1)