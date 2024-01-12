/**
 * @file forwar_functor.cu
 * @author cmcandy
 * @brief
 * @version 0.1
 * @date 2024-01-13
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "kernel/utils/cuda_utils.h"

#include "./forward_functor.h"
#include "./vector_add_1d.cuh"
// kernel launch
namespace kernel
{
    namespace vector_add_1d
    {

        namespace functor
        {
            template <typename DType>
            bool VectorAdd1DOpExecute<DType>::operator()(
                const DType *a_tensor_ptr, const DType *b_tensor_ptr, int length, DType *output_ptr,
                cudaStream_t &stream)
            {
                // set block grid
                dim3 blockSize(256);
                dim3 gridSize((length + blockSize.x - 1) / blockSize.x);
                // launch kernel
                cuda_kernels::vector_add_1d_kernel<DType><<<gridSize, blockSize, 0, stream>>>(a_tensor_ptr, b_tensor_ptr, output_ptr, length);
                return true;
            }
            template struct VectorAdd1DOpExecute<float>;

        } // functor
    }     // vector_add_1d
} // kernel
