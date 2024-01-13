/* Copyright 2021 The TensorPilot Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/

#ifndef UTILS_CONV_UTILS_H_
#define UTILS_CONV_UTILS_H_

#include <vector>
using std::vector;

namespace conv_utils {

template <typename DType>
vector<DType> ComputeConvolutionOutputSpatialShape(
    const vector<DType>& input_spatial_shape, const vector<DType>& kernel_sizes,
    const vector<DType>& strides, const vector<DType>& paddings,
    const vector<DType>& dilations) {
  // the output spatial shape
  vector<DType> output_spatial_shape;

  // compute the output spatial shapes.
  int rank = input_spatial_shape.size();
  for (int i = 0; i < rank; i++) {
    DType out_shape = (input_spatial_shape[i] + 2 * paddings[i] -
                       (dilations[i] * (kernel_sizes[i] - 1) + 1)) /
                          strides[i] +
                      1;
    output_spatial_shape.push_back(out_shape);
  }

  return output_spatial_shape;
}

}  // namespace conv_utils

#endif  // !UTILS_CONV_UTILS_H_
