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
namespace vector_add_1d {

namespace functor {
template <typename DType>
struct VectorAdd1DOpExecute {
  // 重载() 操作符
  bool operator()(const DType *a_tensor_ptr, const DType *b_tensor_ptr,
                  int length, DType *output_ptr, cudaStream_t &stream);
};

}  // namespace functor
}  // namespace vector_add_1d
}  // namespace kernel
