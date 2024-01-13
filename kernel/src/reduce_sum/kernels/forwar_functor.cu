/**
 * @file forwar_functor.cu
 * @author cmcandy
 * @brief
 * @version 0.1
 * @date 2024-01-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "./forward_functor.h"
#include "./reduce_sum_1d.cuh"
#include "kernel/utils/cuda_utils.h"
// kernel launch
namespace kernel {
namespace reduce_sum {

namespace functor {
template <typename DType>
bool ReduceSum1DOpExecute<DType>::operator()(const DType *in_tensor_ptr,
                                             DType *part_output_ptr,
                                             DType *output_ptr, int length,
                                             cudaStream_t &stream) {
  // set block grid
  const int BLOCKSIZE = 256;
  dim3 blockSize(BLOCKSIZE);
  const int GRIDSIZE = (length + blockSize.x - 1) / blockSize.x;
  dim3 gridSize(GRIDSIZE);

  // launch kernel
  cuda_kernels::reduce_sum_1d_kernel<DType, BLOCKSIZE>
      <<<gridSize, blockSize, 0, stream>>>(in_tensor_ptr, part_output_ptr, length);
  cuda_kernels::reduce_sum_1d_kernel<DType, BLOCKSIZE>
      <<<1, blockSize, 0, stream>>>(part_output_ptr, output_ptr, GRIDSIZE);
  return true;
}
template struct ReduceSum1DOpExecute<float>;

}  // namespace functor
}  // namespace reduce_sum
}  // namespace kernel
