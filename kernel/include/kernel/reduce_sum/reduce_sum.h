/**
 * @file    reduce_sum.h
 * @author  cmcandy
 * @brief
 * @version 0.1
 * @date 2024-01-13
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include "kernel/customop.h"

namespace kernel {
namespace reduce_sum {
template <typename DType>
class reduceSumOP : public CustomOpBase {
 public:
  reduceSumOP(int block_size_);
  virtual ~reduceSumOP();
  OP_Status Compute(context::CustomOpContext *context) override;

 protected:
  int block_size_;
};

}  // namespace reduce_sum
}  // namespace kernel
