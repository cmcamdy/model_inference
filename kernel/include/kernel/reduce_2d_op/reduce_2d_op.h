/**
 * @file reduce_op_2d.h
 * @author cmcandy
 * @brief 
 * @version 0.1
 * @date 2024-01-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "kernel/customop.h"

namespace kernel {
namespace  reduce_2d_op{
template <typename DType>
class reduceOP2D : public CustomOpBase {
 public:
  reduceOP2D(int dim, int kernel_type, std::vector<int>& shapes);
  reduceOP2D(int dim, int kernel_type, std::vector<int>&& shapes);
  virtual ~reduceOP2D();
  OP_Status Compute(context::CustomOpContext *context) override;

 protected:
  int dim_;
  int kernel_type_;
  std::vector<int> shapes_;
};

}  // namespace reduce_sum
}  // namespace kernel
