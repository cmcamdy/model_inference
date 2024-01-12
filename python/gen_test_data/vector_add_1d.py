import numpy as np
import os


ops_test_data_dir = os.getenv('OPS_TEST_DATA_DIR')

# 创建一些数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 将数组保存到.npz文件中
np.savez(os.path.join(ops_test_data_dir, 'vector_add_1d_cuda_test.npz'), a=a, b=b)
