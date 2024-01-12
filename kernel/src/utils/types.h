
#pragma once

#include <iostream>
#include <type_traits>
#include "kernel/context/context.h"
// #include "kernel/utils/eigen_utils.h"

namespace kernel
{

    template <typename T>
    DataType ConvertToDataType()
    {
        DataType type = DataType::DT_INT8;
        if (std::is_same<int8_t, T>::value)
        {
            type = DataType::DT_INT8;
        }
        if (std::is_same<uint8_t, T>::value || std::is_same<char, T>::value || std::is_same<bool, T>::value)
        {
            type = DataType::DT_UINT8;
        }
        if (std::is_same<int16_t, T>::value)
        {
            type = DataType::DT_INT16;
        }
        if (std::is_same<uint16_t, T>::value)
        {
            type = DataType::DT_UINT16;
        }
        if (std::is_same<int32_t, T>::value || std::is_same<int, T>::value)
        {
            type = DataType::DT_INT32;
        }
        if (std::is_same<uint32_t, T>::value)
        {
            type = DataType::DT_UINT32;
        }
        if (std::is_same<int64_t, T>::value)
        {
            type = DataType::DT_INT64;
        }
        if (std::is_same<uint64_t, T>::value)
        {
            type = DataType::DT_UINT64;
        }
        if (std::is_same<long long, T>::value)
        {
            type = DataType::DT_INT64;
        }
        // if (std::is_same<Eigen::half, T>::value) {
        //     type = DataType::DT_HALF;
        // }
        if (std::is_same<float, T>::value)
        {
            type = DataType::DT_FLOAT;
        }
        if (std::is_same<double, T>::value)
        {
            type = DataType::DT_DOUBLE;
        }
        return type;
    }

} // namespace kernel
