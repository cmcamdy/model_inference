import numpy as np
import os


ops_test_data_dir = os.getenv('OPS_TEST_DATA_DIR')


# 创建一些数组
in_tensor = np.random.rand(256, 2048).astype(np.float32)
out_tensor_0 = in_tensor.sum(0)
out_tensor_1 = in_tensor.sum(-1)
# out_tensor = in_tensor.sum(0)
# print(out_tensor_0)
# print(out_tensor_1)
print(out_tensor_0.shape)
print(out_tensor_1.shape)
# 将数组保存到.npz文件中
np.savez(os.path.join(ops_test_data_dir, 'reduce_sum_2d_cuda_test.npz'), in_tensor=in_tensor, out_tensor_0=out_tensor_0, out_tensor_1=out_tensor_1)


