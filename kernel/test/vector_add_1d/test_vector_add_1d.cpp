#include <iostream>
#include <gtest/gtest.h>

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

    // ASSERT_EQ(status, kernel::OP_Code::OP_OK) << "submit compute task fail";

    // cudaDeviceSynchronize();
    // auto err = cudaGetLastError();
    // ASSERT_EQ(err, cudaSuccess) << "cuda error, number = " << err
    //                             << " << message = " << cudaGetErrorString(err);
}