import torch
import torch.cuda.nvtx as nvtx

# 初始化CUDA设备
torch.cuda.init()
torch.cuda.synchronize()  # 确保设备就绪

# 定义矩阵尺寸
M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device='cuda')
B = torch.randn(K, N, device='cuda')
C = torch.zeros(M, N, device='cuda')

# 预热运行（排除初始化开销）
C = A @ B
torch.cuda.synchronize()

# 开始性能分析范围
nvtx.range_push("GEMM_Kernel")
torch.cuda.profiler.start()

# 执行多次以获取稳定数据
for _ in range(10):
    C = A @ B
    torch.cuda.synchronize()  # 确保每次迭代完成

# 结束分析范围
torch.cuda.profiler.stop()
nvtx.range_pop()

# 验证结果
print(f"C shape: {C.shape}, sum: {C.sum().item()}")