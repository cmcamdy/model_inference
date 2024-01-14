/**
 * ************************************************************
 * @file          reduce_sum.cc
 * @author        cmcandy
 * @brief
 * @version       0.1
 * @date          2024-01-14
 * @copyright     Copyright (c) 2024
 * ************************************************************
 */

#include "kernel/reduce_sum/reduce_sum.h"

#include <cuda_runtime.h>

#include <vector>

#include "../utils/types.h"
#include "./kernels/forward_functor.h"

namespace kernel {
namespace reduce_sum {
using Tensor = context::ContextTensor;

template <typename DType>
reduceSumOP<DType>::reduceSumOP(int length_)
    : CustomOpBase(*this), length_(length_) {}

template <typename DType>
reduceSumOP<DType>::~reduceSumOP() {}

template <typename DType>
OP_Status reduceSumOP<DType>::Compute(context::CustomOpContext *context) {
  std::cout << "reduceSumOP<DType>::Compute(context::CustomOpContext *context)"
            << std::endl;
  cudaStream_t stream = static_cast<cudaStream_t>(context->GetCudaStream());
  const auto &in_tensor = context->GetInput(0);
  CHECK_COND(in_tensor->dims() == 1, "dim of value_tensor must be 1!");

  const DType *in_tensor_ptr = in_tensor->data<DType>();

  std::vector<std::vector<int64_t>> in_shapes = context->GetInputShapes();

  // reduce sum 1d 的话就是1
  const int BLOCKSIZE = 256;

  auto output_tensor = context->AllocateOutput(
      0, {1}, ConvertToDataType<DType>(), context::DeviceType::CUDA_DEVICE);
  auto part_output_tensor = context->AllocateOutput(
      1, {(in_shapes[0][0] + BLOCKSIZE - 1) / BLOCKSIZE},
      ConvertToDataType<DType>(), context::DeviceType::CUDA_DEVICE);
  DType *part_output_ptr = static_cast<DType *>(part_output_tensor->raw_data);
  DType *output_ptr = static_cast<DType *>(output_tensor->raw_data);

  bool compute_status = functor::ReduceSum1DOpExecute<DType>()(
      in_tensor_ptr, part_output_ptr, output_ptr, length_, stream);
  if (compute_status == true) {
    return OP_Status::OP_OK;
  } else {
    return OP_Status::OP_RUNNING_ERR;
  }
}

template class reduceSumOP<float>;

}  // namespace reduce_sum
}  // namespace kernel