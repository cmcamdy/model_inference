/**
 * *****************************************************************************
 * @file        customop.h
 * @brief
 * @author      cmcandy
 * @date        2023-12-18
 * @copyright   (c) Institute of Computer Technology
 * *****************************************************************************
 */
#pragma once
#include <kernel/context/context.h>

namespace kernel
{

    /*! \brief Custom operator base class*/
    class CustomOpBase
    {
    public:
        CustomOpBase() = delete;
        CustomOpBase(const CustomOpBase &) = default;
        virtual ~CustomOpBase() = default;
        virtual OP_Status Compute(contest::CustomOpContext* context) = 0;
    };
} // kernel