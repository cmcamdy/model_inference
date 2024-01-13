#include <gtest/gtest.h>
#include <math.h>

#include <iostream>

#include "../data_loader/cnpy.h"
#include "../test_context/test_context.h"
#include "../utils.h"
#include "kernel/add/add.h"
#include "kernel/customop.h"
#include "kernel/reduce_sum/reduce_sum.h"
using namespace std;

template <typename T>
std::vector<int64_t> to_shape(T &shape) {
  std::vector<int64_t> shape1;

  shape1.resize(shape.size());

  for (int i = 0; i < shape.size(); ++i) {
    shape1[i] = shape[i];
  }

  return shape1;
}

TEST(reduceSumOP, PositiveNumbers) {
  using DType = float;
  const std::string test_data_path = kernel::test::GetTestDataPath();
  auto context = std::make_unique<kernel::test::TestContext>();
  std::cout << test_data_path << std::endl;
  std::string file = test_data_path + "/reduce_sum_cuda_test.npz";
  auto kvs = cnpy::npz_load(file);

  auto in = kvs["in_tensor"];
  // check shape
  std::cout << in.shape[0] << std::endl;

  auto in_tensor =
      context->MakeTensor<DType>(to_shape(in.shape), in.data<DType>());

  // set input
  context->SetInput(0, in_tensor);
  // outputs
  context->SetOutputType(0, kernel::test::DataTypeToEnum<DType>::value);
  context->SetOutputType(1, kernel::test::DataTypeToEnum<DType>::value);

  // build instance
  auto inst =
      std::make_unique<kernel::reduce_sum::reduceSumOP<DType>>(in.shape[0]);
  auto status = inst->Compute(context.get());
  cudaDeviceSynchronize();
  std::cout << "status:" << (status == kernel::OP_Code::OP_OK) << std::endl;

  auto out_host = context->ToHostTensor(context->GetOutput(0));
  auto out_true = kvs["out_tensor"];

  int counter = 0;

  int num_elements = 1;
  for (int i = 0; i < out_host->shape.size(); ++i) {
    num_elements *= out_host->shape[i];
  }
  std::vector<float> output_data_float(num_elements);
  for (int i = 0; i < num_elements; i++) {
    output_data_float[i] = (float)(out_host->data<DType>()[i]);
  }

  std::cout << num_elements << " elements for output" << std::endl;
  DType maxDiff = 0;

  for (uint32_t i = 0; i < num_elements; ++i) {
    DType c = out_host->data<DType>()[i];
    DType t = out_true.data<DType>()[i];
    DType diff = c - t;
    maxDiff = std::max(maxDiff, abs(diff));
    std::cout << c << " " << i << ":" << diff << " diff" << std::endl;

    // TODO(owner): check method
    EXPECT_TRUE(fabs(diff) < 0.022)
        << "for output, the " << (counter++, i)
        << "th data not match, diff = " << diff << ", cal: " << c
        << ", target: " << t << ", output variable: output with DType";
  }
  printf("max err= %f\n", maxDiff);

  context->Reset();
}