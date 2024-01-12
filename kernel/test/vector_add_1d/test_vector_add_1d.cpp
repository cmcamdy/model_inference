#include <iostream>
#include <gtest/gtest.h>
#include <math.h>

#include "../test_context/test_context.h"
#include "../data_loader/cnpy.h"
#include "../utils.h"

#include "kernel/vector_add_1d/vector_add_1d.h"
#include "kernel/customop.h"
#include "kernel/add/add.h"
using namespace std;

template <typename T>
std::vector<int64_t> to_shape(T &shape)
{
    std::vector<int64_t> shape1;

    shape1.resize(shape.size());

    for (int i = 0; i < shape.size(); ++i)
    {
        shape1[i] = shape[i];
    }

    return shape1;
}

TEST(VectorAdd1DOp, PositiveNumbers)
{
    // EXPECT_EQ(add(1, 2), 3);
    // EXPECT_EQ(add(5, 7), 12);
    using DType = float;
    const std::string test_data_path = kernel::test::GetTestDataPath();
    auto context = std::make_unique<kernel::test::TestContext>();

    std::cout << "helloworld~" << std::endl;
    std::cout << test_data_path << std::endl;
    std::string file = test_data_path + "/vector_add_1d_cuda_test.npz";
    auto kvs = cnpy::npz_load(file);

    // a
    auto a = kvs["a"];
    // shape check, see api here: kernel/test/data_loader/cnpy.h
    // for (auto s : a.shape)
    // {
    //     cout << s << endl;
    // }
    // for(int i=0; i<a.shape[0]; ++i){
    //     cout << a.data<DType>()[i] << endl;
    // }
    auto a_tensor = context->MakeTensor<DType>(to_shape(a.shape), a.data<DType>());

    // b
    auto b = kvs["b"];
    auto b_tensor = context->MakeTensor<DType>(to_shape(b.shape), b.data<DType>());

    // set input
    context->SetInput(0, a_tensor);
    context->SetInput(1, b_tensor);
    // outputs
    context->SetOutputType(0, kernel::test::DataTypeToEnum<DType>::value);

    // build instance
    auto inst = std::make_unique<kernel::vector_add_1d::VectorAdd1DOp<DType>>(3);

    auto status = inst->Compute(context.get());
    cudaDeviceSynchronize();
    std::cout << "status:" << (status == kernel::OP_Code::OP_OK) << std::endl;
    // ASSERT_EQ(status, kernel::OP_Code::OP_OK) << "submit compute task fail";
    // EXPECT_EQ(status, kernel::OP_Code::OP_OK);
    // 这个可以，但是上面这个就不行
    // EXPECT_EQ(add(1, 2), 3);
    // ASSERT_EQ(add(1, 2), 3) << "submit compute task fail";

    auto out = context->GetOutput(0);
    auto out_host = context->ToHostTensor(out);
    auto out_true = kvs["c"];
    int counter = 0;

    int num_elements = 1;
    for (int i = 0; i < out_host->shape.size(); ++i)
    {
        num_elements *= out_host->shape[i];
    }

    std::vector<float> output_data_float(num_elements);
    for (int i = 0; i < num_elements; i++)
    {
        output_data_float[i] = (float)(out_host->data<DType>()[i]);
    }

    std::cout << num_elements << " elements for output" << std::endl;

    DType maxDiff = 0;

    for (uint32_t i = 0; i < num_elements; ++i)
    {

        DType c = out_host->data<DType>()[i];
        DType t = out_true.data<DType>()[i];
        DType diff = c - t;
        maxDiff = std::max(maxDiff, abs(diff));
        std::cout << i << ":" << diff << " diff" << std::endl;

        // TODO(owner): check method
        EXPECT_TRUE(fabs(diff) < 0.022)
            << "for output, the " << (counter++, i) << "th data not match, diff = " << diff
            << ", cal: " << c << ", target: " << t << ", output variable: output with DType";
    }
    printf("max err= %f\n", maxDiff);

    context->Reset();
}