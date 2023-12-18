/**
 * *****************************************************************************
 * @file        context.h
 * @brief       
 * @author      cmcandy
 * @date        2023-12-18
 * @copyright   (c) Institute of Computer Technology
 * *****************************************************************************
 */
#pragma once
#include <kernel/utils/error.h>

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace kernel
{

    typedef enum class OP_Code : unsigned char
    {
        OP_OK = 0,
        OP_INVAILD_ARGUMENT = 1,
        OP_UNKNOWN = 2,
        OP_CHECK_ERR = 3,
        OP_OUT_OF_RANGE = 4,
        OP_NULL_POINTER = 5,
        OP_UNSUPPORTED_ERR = 6,
        OP_RUNNING_ERR = 7
    } OP_Code;
    typedef OP_Code OP_Status;

    /*! @brief The data type definition. */
    enum class DataType : uint8_t
    {
        DT_INT8 = 0x1,
        DT_UINT8,

        DT_INT16,
        DT_UINT16,

        DT_INT32,
        DT_UINT32,

        DT_INT64,
        DT_UINT64,

        DT_HALF,
        DT_FLOAT,
        DT_DOUBLE
    };

    inline int64_t length(const std::vector<int64_t> &shape)
    {
        if (shape.size() > 0)
        {
            int64_t len = shape[0];
            for (size_t i = 1; i < shape.size(); i++)
            {
                len *= shape[i];
            }
            return len;
        }
        return 0;
    }

    namespace context
    {

        enum DeviceType
        {
            CPU_DEVICE = 0,
            CUDA_DEVICE = 1,
            CUDA_HOST_DEVICE = 2
        };

        typedef void *ContextStreamHandle;
        typedef void *ContextEventHandle;

        /*!
         * \brief This is designed as a Tensor wrapper. Some notice:
         * 1. As a wrapper, ContextTensor do not own the `raw_data` memory, users need to manager the
         * memory buffer on their own.
         * 2. `raw_shape` and `raw_dim` must be used together, ContextTensor also do not own the
         * `raw_shape` buffer. Or user can just use `shape`.
         */
        struct ContextTensor
        {
            /*! \brief The raw data pointer of this tensor. */
            void *raw_data{nullptr};
            /*! \brief The raw tensor pointer */
            void *raw_tensor{nullptr};

            /*! \brief The raw shape pointer of this tensor. */
            int64_t *raw_shape{nullptr};
            /*! \brief The shape dim if raw_shape is used. */
            int32_t raw_dim{-1};

            /*! \brief The the tensor data type. */
            DataType data_type{DataType::DT_INT8};
            /*! \brief The device identifier. */
            DeviceType device_type{DeviceType::CPU_DEVICE};

            /*! \brief The shape of this tensor, if `raw_shape` is not used, this will be used. */
            std::vector<int64_t> shape;

            /*! \brief Api similar to tf. */
            template <typename DType>
            inline DType *data() const
            {
                return reinterpret_cast<DType *>(raw_data);
            }
            inline int32_t dims() const { return raw_dim != -1 ? raw_dim : shape.size(); }
            inline int64_t dim_size(int32_t i) const
            {
                if (i >= dims())
                {
                    return 0;
                }
                return raw_shape ? raw_shape[i] : shape[i];
            }
            inline int64_t num_elements() const { return length(); }

            inline OP_Status update_shape(const std::vector<int64_t> &new_shape)
            {
                if (dims() != static_cast<int32_t>(new_shape.size()))
                {
                    LOGE("The dims of original tensor should be the same as the new shape.");
                    return OP_Status::OP_INVAILD_ARGUMENT;
                }
                if (num_elements() < kernel::length(new_shape))
                {
                    LOGE("The original tensor size should be larger than the new shape.");
                    return OP_Status::OP_INVAILD_ARGUMENT;
                }

                int64_t *shape_to_update = raw_shape ? raw_shape : shape.data();
                for (size_t i = 0; i < new_shape.size(); i++)
                {
                    shape_to_update[i] = new_shape[i];
                }
                return OP_Status::OP_OK;
            }

            inline std::vector<int64_t> get_shape() const
            {
                return raw_shape ? std::vector<int64_t>(raw_shape, raw_shape + raw_dim) : shape;
            }

            inline int64_t length() const
            {
                if (!raw_shape)
                {
                    return kernel::length(shape);
                }
                if (raw_dim > 0)
                {
                    int64_t len = raw_shape[0];
                    for (int32_t i = 1; i < raw_dim; i++)
                    {
                        len *= raw_shape[i];
                    }
                    return len;
                }
                return 0;
            }
        };

        typedef std::shared_ptr<ContextTensor> TensorPointer;
        typedef std::shared_ptr<ContextTensor> &TensorPointerRef;

        class CustomOpContext
        {
        public:
            CustomOpContext() = default;
            virtual ~CustomOpContext() = default;

            /*! \brief Get cuda stream handle. */
            virtual ContextStreamHandle GetCudaStream() const = 0;

            /*! \brief Get input tensor from this context.
             * \param index The index of the input tensor.
             * \note In tf, we can simply overwrite this impl to use `context->input()`.
             */
            virtual TensorPointer GetInput(size_t index) const = 0;

            /*! @brief Get input tensor number. */
            virtual size_t GetInputSize() const = 0;

            /*! @brief Get input tensor shpaes. */
            virtual std::vector<std::vector<int64_t>> GetInputShapes() const = 0;

            /*! \brief Allocate(For tvm tvm is get) output tensor from this context.
             * \param index The index of the output tensor.
             * \param shape The shape of the output tensor.
             * \param dtype The data type of the output tensor.
             * \param dev   The device type of the output tensor.
             * \return The ContextTensor pointer.
             * \note In tf, we can simply overwrite this impl to use
             * `context->AllocateOutput()`.
             */
            virtual TensorPointer AllocateOutput(int32_t index, const std::vector<int64_t> &shape,
                                                 DataType dtype = DataType::DT_INT8,
                                                 DeviceType dev = DeviceType::CPU_DEVICE) = 0;

            /*! \brief Allocate temp buffer, will be released.
             * \param size The total size to allocate, in bytes.
             * \param dtype The data type to allocate buffer.
             * \param dev The target device to allocate buffer on.
             */
            virtual void *Allocate(int64_t size, DataType dtype = DataType::DT_INT8,
                                   DeviceType dev = DeviceType::CPU_DEVICE) = 0;

            /*! \brief Deallocate temp tensors buffer. will be called before destructor.
             *         The resouces should be released automatically before context exit.
             *  \param ptr The temp tensor buffer pointer.
             */
            virtual bool DeAllocate(void *ptr) = 0;

            /*! \brief Slice the given tensor and return new tensor.
             * \param tensor The target sliced tensor.
             * \param start The dims0 start position.
             * \param size The slice size.
             * \return The sliced tensor.
             */
            virtual TensorPointer Slice(TensorPointer tensor, int start, int size) = 0;

            /*! \brief Copy the input tensor to output based on index, return the output tensor.
             * \param output_idx The output tensor index.
             * \param input_idx The input tensor index.
             * \return The output Tensor.
             */
            virtual TensorPointer SetOutput(int output_idx, int input_idx) = 0;

            /*! \brief Get user-data struct pointer. */
            virtual void *GetUD() = 0;

            /*! \brief Return shared memory available per block in bytes of devices. */
            virtual size_t GetDeviceSharedMemPerBlock() = 0;
        };

    } // namespace context
} // namespace kernel
