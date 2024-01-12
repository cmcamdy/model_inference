/**
 * @file vector_add_1d.cuh
 * @author cmcandy
 * @brief
 * @version 0.1
 * @date 2024-01-13
 *
 * @copyright Copyright (c) 2024
 *
 */

namespace kernel
{
    namespace vector_add_1d
    {

        namespace functor
        {

            namespace cuda_kernels
            {
                template <typename DType>
                __global__ void vector_add_1d_kernel(const DType *a_tensor_ptr, const DType *b_tensor_ptr, DType *c_tensor_ptr, int length)
                {
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if(idx < length){
                        c_tensor_ptr[idx] = a_tensor_ptr[idx] + b_tensor_ptr[idx];
                    }
                }
            } // cuda_kernels

        } // functor
    }     //  vector_add_1d
} // kernel
