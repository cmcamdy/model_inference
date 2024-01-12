#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <list>
#include <map>

// #include "eigen3/Eigen/Dense"
#include "kernel/context/context.h"

namespace kernel
{
  namespace test
  {

    using ContextStreamHandle = context::ContextStreamHandle;
    using ContextTensor = context::ContextTensor;
    using TensorPointer = context::TensorPointer;
    using DeviceType = context::DeviceType;

    template <typename DType>
    struct DataTypeToEnum
    {
      const static DataType value = DataType::DT_INT8;
    };

    template <>
    struct DataTypeToEnum<int8_t>
    {
      const static DataType value = DataType::DT_INT8;
    };

    template <>
    struct DataTypeToEnum<uint8_t>
    {
      const static DataType value = DataType::DT_UINT8;
    };

    template <>
    struct DataTypeToEnum<int16_t>
    {
      const static DataType value = DataType::DT_INT16;
    };

    template <>
    struct DataTypeToEnum<uint16_t>
    {
      const static DataType value = DataType::DT_UINT16;
    };

    template <>
    struct DataTypeToEnum<int32_t>
    {
      const static DataType value = DataType::DT_INT32;
    };

    template <>
    struct DataTypeToEnum<uint32_t>
    {
      const static DataType value = DataType::DT_UINT32;
    };

    template <>
    struct DataTypeToEnum<int64_t>
    {
      const static DataType value = DataType::DT_INT64;
    };

    template <>
    struct DataTypeToEnum<uint64_t>
    {
      const static DataType value = DataType::DT_UINT64;
    };

    // template <>
    // struct DataTypeToEnum<Eigen::half>
    // {
    //   const static DataType value = DataType::DT_HALF;
    // };

    // template <>
    // struct DataTypeToEnum<__half>
    // {
    //   const static DataType value = DataType::DT_HALF;
    // };

    template <>
    struct DataTypeToEnum<float>
    {
      const static DataType value = DataType::DT_FLOAT;
    };

    template <>
    struct DataTypeToEnum<double>
    {
      const static DataType value = DataType::DT_DOUBLE;
    };

    class FreelistAllocator
    {
    public:
      FreelistAllocator() = default;
      ~FreelistAllocator() = default;

      void Manage(void *data, size_t size, const char *annotation,
                  bool with_guard = false, cudaStream_t stream = nullptr);
      void *Allocate(size_t size);
      void Free(void *ptr);

      void Stats();
      void Reset();
      void SetGuard();
      void CheckGuard();

    private:
      /*
       * begin      head                 tail     end
       * |-----------|====================|-------|
       * |   guard   |        data        | guard |
       */
      const int prefix_guard_length = 1024;
      const int suffix_guard_length = 2048;
      typedef struct
      {
        uint64_t begin;
        uint64_t end;
        uint64_t head;
        uint64_t tail;
      } Block;

      const int align_bytes = 16;

      uint8_t *raw_base_; // it maybe not aligend to align_bytes
      uint8_t *base_;     // we have aligend it to align_bytes

      size_t raw_size_;
      size_t size_;

      cudaStream_t stream_;
      bool with_guard_ = false;

      std::list<Block> alloc_;
      std::list<Block> free_;

      std::string annotation_;
    };

    class TestContext : public context::CustomOpContext
    {
    public:
      TestContext();
      ~TestContext();

      ContextStreamHandle GetCudaStream() const override;

      TensorPointer GetInput(size_t index) const override;

      size_t GetInputSize() const override;

      virtual std::vector<std::vector<int64_t>> GetInputShapes() const override;

      TensorPointer AllocateOutput(
          int32_t index, const std::vector<int64_t> &shape,
          DataType dtype = DataType::DT_INT8,
          DeviceType dev = DeviceType::CPU_DEVICE) override;

      void *Allocate(int64_t size, DataType dtype = DataType::DT_INT8,
                     DeviceType dev = DeviceType::CUDA_DEVICE) override;

      bool DeAllocate(void *ptr) override;

      TensorPointer Slice(TensorPointer tensor, int start, int size) override;

      TensorPointer SetOutput(int output_idx, int input_idx) override;

      void *GetUD() override;

      size_t GetDeviceSharedMemPerBlock() override;

      TensorPointer MakeTensor(const std::vector<int64_t> &shape,
                               DataType data_type, const void *host_data = nullptr,
                               DeviceType device = DeviceType::CUDA_DEVICE);

      template <typename DType>
      TensorPointer MakeTensor(const std::vector<int64_t> &shape,
                               const void *host_data = nullptr,
                              //  DeviceType device = DeviceType::CPU_DEVICE)
                               DeviceType device = DeviceType::CUDA_DEVICE)
      {
        DataType data_type = DataTypeToEnum<DType>::value;
        return MakeTensor(shape, data_type, host_data, device);
      }

      TensorPointer ToHostTensor(TensorPointer);

      void SetInput(int index, TensorPointer tensor);

      void SetOutputType(int index, DataType type);

      void SetOutputDevice(int index, DeviceType type);

      TensorPointer GetOutput(int index, DataType *data_type = nullptr,
                              DeviceType *device_type = nullptr);

      void Stats();

      // reset memory
      void Reset();

    private:
      cudaStream_t stream_;
      std::map<int, TensorPointer> inputs_;
      std::map<int, TensorPointer> outputs_;
      std::map<int, DataType> output_data_types_;
      std::map<int, DeviceType> output_device_types_;

      // memory pointer
      void *cuda_memory_;
      void *host_memory_;
      void *pinned_memory_;

      const int64_t cuda_memory_size_ = 1ull << 32; // 1GB
      const int64_t host_memory_size_ = 1ull << 32; // 1GB
      const int64_t pinned_memory_size_ = 1 << 29;  // 500MB

      bool memory_check_mode_;

      // internal memory pool
      FreelistAllocator cuda_allocator_;
      FreelistAllocator host_allocator_;
      FreelistAllocator pinned_allocator_;
    };

  } // namespace test
} // namespace kernel
