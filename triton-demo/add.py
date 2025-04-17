import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, Z, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets)
    y = tl.load(Y + offsets)
    z = x + y
    tl.store(Z + offsets, z)

# 创建一些示例数据
import torch

size = 1024
X = torch.randn(size, device='cuda')
Y = torch.randn(size, device='cuda')
Z = torch.empty(size, device='cuda')

# 调用内核
grid = (size // 1024 + 1,)
add_kernel[grid](X, Y, Z, BLOCK_SIZE=1024)

# 验证结果
assert torch.allclose(Z, X + Y)
print("Triton kernel executed successfully!")
# https://zhuanlan.zhihu.com/p/887257776