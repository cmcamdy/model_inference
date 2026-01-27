import math
import os
import torch
import torch.nn as nn
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
custom_op = load(name='custom_op', sources=['main.cpp', 'embedding.cu'], extra_cuda_cflags=['-O2'])


vocab_size = 10000  # 符号总数（比如1万个单词）
embedding_dim = 128  # 嵌入向量维度
embedding = nn.Embedding(
    num_embeddings=vocab_size,  # 符号总数V
    embedding_dim=embedding_dim,  # 向量维度D
    padding_idx=None  # 可选，指定填充符号的索引（比如0），其向量不更新
)

# 2. 准备输入（离散符号的索引）
# 假设输入是一批样本，每个样本包含3个符号索引
input_indices = torch.tensor([[10, 20, 30], [40, 50, 60]])  # shape: (2, 3)

# 3. 计算Embedding（核心：查表）
output_embeddings = embedding(input_indices)

# 输出形状：(2, 3, 128) → 2个样本，每个样本3个符号，每个符号128维向量
print(output_embeddings.shape)  # torch.Size([2, 3, 128])

# 4. 验证：直接取Embedding矩阵的第10行，和输入索引10的结果一致
embedding_matrix = embedding.weight  # 获取Embedding矩阵（shape: 10000, 128）
# print(torch.equal(embedding_matrix[10], output_embeddings[0, 0]))  # True

weight = torch.randn(100, 128)
print(embedding_matrix)
print(input_indices.reshape(-1).int())
custom_y = custom_op.embedding_naive(input_indices.reshape(-1).int(), weight, 1)
# print(custom_y)
print(custom_y.shape)

# print('values sanity check :', torch.allclose(torch.max((custom_y - output_embeddings)), torch.zeros(1).cuda(), rtol=1e-05, atol=1e-05))
