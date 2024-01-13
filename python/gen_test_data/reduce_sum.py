import numpy as np
import os


ops_test_data_dir = os.getenv('OPS_TEST_DATA_DIR')

arr = [index * 1.0 for index in range(256)]
# 创建一些数组
in_tensor = np.array(arr).astype(np.float32)
out_tensor = in_tensor.sum()
print(out_tensor)
# 将数组保存到.npz文件中
np.savez(os.path.join(ops_test_data_dir, 'reduce_sum_cuda_test.npz'), in_tensor=in_tensor, out_tensor=out_tensor)
