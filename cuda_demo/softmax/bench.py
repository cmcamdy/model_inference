import math
import os
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# 3. PyTorch CPU随机数
torch.manual_seed(42)
# 4. PyTorch CUDA随机数（单GPU）
torch.cuda.manual_seed(42)

# export CC=/usr/bin/gcc-9
# export CXX=/usr/bin/g++-9
# Load the CUDA kernel as a python module
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # 根据你的 GPU 架构调整
custom_op = load(name='custom_op', sources=['main.cpp', 'softmax.cu', 'softmax_smem.cu'], extra_cuda_cflags=['-O2'])

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    z = x
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # ret = numerator
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

M,N = 16, 16

# for i in range(4,6):
# for i in range(6,7):
# for i in range(4,11):
# for i in range(10,11):
# for i in range(12,13):
for i in range(13,14):
    M,N = 1<<i, 1<<i
    # M,N = 1, 1<<i
    print(M,N)
    X = torch.randn(M, N).cuda()
    # X = torch.arange(N).float().unsqueeze(dim=0).cuda()
    print('=== profiling transpose ===') 
    # Y = custom_op.softmax_2d_naive(X, 1)
    Y = custom_op.softmax_2d_smem(X, 100)
    # print(X)
    # Y_torch = torch.softmax(X, dim=1)
    # Y_torch = naive_softmax(X)
    # print(Y_torch.shape, Y.shape)
    # print(Y_torch)
    # print(Y)
    # print(torch.max((Y-Y_torch)))
    # print("*"*50)
    # print('values sanity check (M,N):', M, N, torch.allclose(torch.max((Y-Y_torch)), torch.zeros(1).cuda(), rtol=1e-05, atol=1e-05))
