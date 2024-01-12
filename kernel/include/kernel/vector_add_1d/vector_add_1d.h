/**
 * *****************************************************************************
 * @file        vector_add.h
 * @brief
 * @author      cmcandy
 * @date        2023-12-18
 * @copyright   (c) Institute of Computer Technology
 * *****************************************************************************
 */

#pragma once

#include "kernel/customop.h"
namespace kernel
{
    namespace vector_add_1d
    {
        template <typename DType>
        class VectorAdd1DOp : public CustomOpBase
        {
        public:
            VectorAdd1DOp(int vector_len);
            virtual ~VectorAdd1DOp();
            OP_Status Compute(context::CustomOpContext *context) override;

        protected:
            int vector_len_;
        };
        // template class VectorAdd1DOp<float>;
    } // namespace vector_add_1d
} // namespace kernel