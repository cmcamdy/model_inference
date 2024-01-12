#include <cuda_runtime.h>
#include <vector>

#include "kernel/vector_add_1d/vector_add_1d.h"
#include "./kernels/forward_functor.h"
#include "../utils/types.h"

namespace kernel
{

    namespace vector_add_1d
    {
        using Tensor = context::ContextTensor;

        template <typename DType>
        VectorAdd1DOp<DType>::VectorAdd1DOp(int vector_len)
            : CustomOpBase(*this),
              vector_len_(vector_len)
        {
        }

        template <typename DType>
        VectorAdd1DOp<DType>::~VectorAdd1DOp() {}

        template <typename DType>
        OP_Status VectorAdd1DOp<DType>::Compute(context::CustomOpContext *context)
        {
            // =================== function un-related to inputs begin ================ //
            cudaStream_t stream = static_cast<cudaStream_t>(context->GetCudaStream());
            // gpu allocator function
            // =================== function un-related to inputs ends ================ //

            const auto &a_tensor = context->GetInput(0);
            const auto &b_tensor = context->GetInput(1);
            // std::cout << "a_tensor dims:" << a_tensor->dims() << std::endl;
            // std::cout << "a_tensor shape:" << a_tensor->get_shape()[0] << std::endl;
            // const auto data = a_tensor->data<DType>();
            // // std::cout << "a_tensor data:" << data << std::endl;
            // std::cout << "a_tensor typeid:" << typeid(a_tensor).name() << std::endl;
            // std::cout << "a_tensor typeid:" << typeid(data).name() << std::endl;

            CHECK_COND(a_tensor->dims() == 1 && b_tensor->dims() == 1 && a_tensor->dim_size(0) == b_tensor->dim_size(0),
                       "dim of value_tensor must be 1, shape of a_tensor = b_tensor!");

            const DType *a_tensor_ptr = a_tensor->data<DType>();
            const DType *b_tensor_ptr = b_tensor->data<DType>();
            // 打印不出来是因为cudaMem的原因，如果context的device是cpu，那么是可以打印出来数据的
            // std::cout << "a_tensor_ptr data:" << a_tensor_ptr[0] << std::endl;

            std::vector<std::vector<int64_t>> in_shapes = context->GetInputShapes();

            auto output_tensor = context->AllocateOutput(0, in_shapes[0], ConvertToDataType<DType>(),
                                                         context::DeviceType::CUDA_DEVICE);
            DType *output_ptr = static_cast<DType *>(output_tensor->raw_data);
            bool compute_status = functor::VectorAdd1DOpExecute<DType>()(a_tensor_ptr, b_tensor_ptr, in_shapes[0][0], output_ptr, stream);
            if (compute_status == true)
            {
                return OP_Status::OP_OK;
            }
            else
            {
                return OP_Status::OP_RUNNING_ERR;
            }
        }

        template class VectorAdd1DOp<float>;

    } // namespace vector_add_1d
} // namespace kernel