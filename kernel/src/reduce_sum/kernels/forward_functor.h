/**
 * @file forward_functor.h
 * @author cmcandy
 * @brief
 * @version 0.1
 * @date 2024-01-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <functional>

#include "kernel/utils/cuda_utils.h"

namespace kernel {
namespace reduce_sum {

namespace functor {
template <typename DType>
struct ReduceSum1DOpExecute {
  // 重载() 操作符
  bool operator()(const DType *in_tensor_ptr, DType *part_output_ptr,
                  DType *output_ptr, int length, cudaStream_t &stream);
};

}  // namespace functor
}  // namespace reduce_sum
}  // namespace kernel
