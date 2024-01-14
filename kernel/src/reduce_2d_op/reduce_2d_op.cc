/**
 * ************************************************************
 * @file          reduce_2d_op.cc
 * @author        cmcandy
 * @brief
 * @version       0.1
 * @date          2024-01-14
 * @copyright     Copyright (c) 2024
 * ************************************************************
 */

#include "kernel/reduce_2d_op/reduce_2d_op.h"

#include <cuda_runtime.h>

#include <vector>

#include "../utils/types.h"
// #include "./kernels/forward_functor.h"

namespace kernel {
namespace reduce_2d_op {
using Tensor = context::ContextTensor;

template <typename DType>
reduceOP2D<DType>::reduceOP2D(int dim, int kernel_type,
                              std::vector<int>& shapes)
    : CustomOpBase(*this),
      dim_(dim),
      kernel_type_(kernel_type),
      shapes_(shapes) {
  // 不确定vector是否能这么初始化
}

template <typename DType>
reduceOP2D<DType>::reduceOP2D(int dim, int kernel_type,
                              std::vector<int>&& shapes)
    : CustomOpBase(*this),
      dim_(dim),
      kernel_type_(kernel_type),
      shapes_(shapes) {
  // 不确定vector是否能这么初始化
}

template <typename DType>
reduceOP2D<DType>::~reduceOP2D() {}

template <typename DType>
OP_Status reduceOP2D<DType>::Compute(context::CustomOpContext* context) {
  std::cout << "reduceOP2D<DType>::Compute(context::CustomOpContext *context)"
            << std::endl;
  std::cout << shapes_[0] << " " << shapes_[1] << std::endl;

  //TODO kernel实现
  
  return OP_Status::OP_OK;
}
template class reduceOP2D<float>;

}  // namespace reduce_2d_op
}  // namespace kernel