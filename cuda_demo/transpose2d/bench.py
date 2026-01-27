import math
import os
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# export CC=/usr/bin/gcc-9
# export CXX=/usr/bin/g++-9
# Load the CUDA kernel as a python module
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # 根据你的 GPU 架构调整
custom_op = load(name='custom_op', sources=['main.cpp', 'transpose.cu'], extra_cuda_cflags=['-O2'])



M,N = 16, 16
M,N = 16, 16

# for i in range(4,6):
# for i in range(5,6):
# for i in range(4,11):
for i in range(12,13):
    M,N = 1<<i, 1<<i
    print(M,N)
    X = torch.randn(M, N).cuda()
    print('=== profiling transpose ===') 
    Y = custom_op.transpose2d_naive(X)
    Y = custom_op.transpose2d_vector(X)
    # print(X.T, Y)
    # print('values sanity check (M,N):', M, N, torch.allclose(X.T.float(), Y, atol=1e-02))
