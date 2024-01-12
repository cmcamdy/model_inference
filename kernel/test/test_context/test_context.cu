#include <cuda_fp16.h>
#include <unistd.h>

#include <stdexcept>

#include "kernel/context/context.h"
#include "test_context.h"

namespace kernel
{
  namespace test
  {

    static uint32_t SizeOfDataType(DataType type)
    {
      switch (type)
      {
      case DataType::DT_INT8:
      case DataType::DT_UINT8:
      {
        return 1;
      }
      case DataType::DT_HALF:
      case DataType::DT_INT16:
      case DataType::DT_UINT16:
      {
        return 2;
      }
      case DataType::DT_FLOAT:
      case DataType::DT_INT32:
      case DataType::DT_UINT32:
      {
        return 4;
      }
      case DataType::DT_DOUBLE:
      case DataType::DT_INT64:
      case DataType::DT_UINT64:
      {
        return 8;
      }
      }
      return 0;
    }

    TestContext::TestContext()
    {
      cudaFree(0);
      cudaStreamCreate(&stream_);

      memory_check_mode_ = false;
      const char *penv1 = getenv("CUDA_MEMCHECK");
      if (penv1)
      {
        memory_check_mode_ = true;
      }
      const char *penv2 = ::getenv("NV_SANITIZER_INJECTION_PORT_RANGE_BEGIN");
      if (penv2)
      {
        memory_check_mode_ = true;
      }
      const char *penv3 = ::getenv("NV_SANITIZER_INJECTION_PORT_BASE");
      if (penv3)
      {
        memory_check_mode_ = true;
      }

      if (memory_check_mode_)
      {
        printf(
            "========= Memory Check Enabled in TestContext, we don't use memory "
            "pool any more!\n");
      }

      cuda_memory_ = nullptr;
      // cuda memory
      if (!memory_check_mode_)
      {
        const int64_t size = cuda_memory_size_;
        void *data = nullptr;
        cudaMallocAsync(&data, size, stream_);
        // touch it to construct the TLB
        cudaMemsetAsync(data, 0, size, stream_);
        cuda_allocator_.Manage(data, size, "cuda", true, stream_);
        cuda_memory_ = data;
      }

      // host memory
      {
        const int64_t size = host_memory_size_;
        void *data = nullptr;
        data = ::malloc(size);
        // touch it to avoid minior-pagefault
        ::memset(data, 0, size);
        host_allocator_.Manage(data, size, "host");
        host_memory_ = data;
      }

      // pinned memory
      {
        const int64_t size = pinned_memory_size_;
        void *data = nullptr;
        cudaMallocHost(&data, size);
        // touch it to avoid minior-pagefault
        ::memset(data, 0, size);
        pinned_allocator_.Manage(data, size, "pinned");
        pinned_memory_ = data;
      }
    }

    TestContext::~TestContext()
    {
      if (!memory_check_mode_ && nullptr != cuda_memory_)
      {
        cuda_allocator_.CheckGuard();
        cudaFreeAsync(cuda_memory_, stream_);
      }

      if (nullptr != host_memory_)
      {
        ::free(host_memory_);
      }

      if (nullptr != pinned_memory_)
      {
        cudaFreeHost(pinned_memory_);
      }
    }

    TensorPointer TestContext::MakeTensor(const std::vector<int64_t> &shape,
                                          DataType data_type, const void *host_data,
                                          DeviceType device)
    {
      uint64_t size = SizeOfDataType(data_type);
      for (const auto &x : shape)
      {
        size *= x;
      }

      void *data = nullptr;
      if (device == DeviceType::CUDA_DEVICE)
      {
        if (!memory_check_mode_)
        {
          data = cuda_allocator_.Allocate(size);
        }
        else
        {
          cudaMallocAsync(&data, size, stream_);
        }
      }
      else if (device == DeviceType::CPU_DEVICE)
      {
        data = host_allocator_.Allocate(size);
      }
      else if (device == DeviceType::CUDA_HOST_DEVICE)
      {
        data = pinned_allocator_.Allocate(size);
      }

      if (host_data)
      {
        if (device == DeviceType::CUDA_DEVICE)
        {
          cudaMemcpyAsync(data, host_data, size, cudaMemcpyHostToDevice, stream_);
        }
        else if (device == DeviceType::CPU_DEVICE)
        {
          memcpy(data, host_data, size);
        }
        else if (device == DeviceType::CUDA_HOST_DEVICE)
        {
          memcpy(data, host_data, size);
        }
      }

      auto tensor = std::make_shared<ContextTensor>();
      tensor->raw_data = data;
      tensor->data_type = data_type;
      tensor->device_type = DeviceType::CUDA_DEVICE;
      tensor->shape = shape;

      return tensor;
    }

    ContextStreamHandle TestContext::GetCudaStream() const
    {
      return static_cast<ContextStreamHandle>(stream_);
    }

    TensorPointer TestContext::GetInput(size_t index) const
    {
      auto it = inputs_.find(index);
      if (it == inputs_.end())
      {
        throw std::runtime_error(
            "please set input first before invoke CustomOpBase::Compute(...)");
      }

      return it->second;
    }

    size_t TestContext::GetInputSize() const { return inputs_.size(); }

    std::vector<std::vector<int64_t>> TestContext::GetInputShapes() const
    {
      size_t input_num = GetInputSize();
      std::vector<std::vector<int64_t>> input_shapes;
      for (size_t i = 0; i < input_num; i++)
      {
        TensorPointer tensor = GetInput(i);
        input_shapes.push_back(tensor->get_shape());
      }
      return input_shapes;
    }

    TensorPointer TestContext::AllocateOutput(int32_t index,
                                              const std::vector<int64_t> &shape,
                                              DataType dtype, DeviceType dev)
    {
      auto itt = output_data_types_.find(index);
      if (itt == output_data_types_.end())
      {
        throw std::runtime_error("output type have not been set");
      }

      DataType data_type = itt->second;

      DeviceType device_type = DeviceType::CUDA_DEVICE;
      auto itd = output_device_types_.find(index);
      if (itd != output_device_types_.end())
      {
        device_type = itd->second;
      }

      TensorPointer tensor = MakeTensor(shape, data_type, nullptr, device_type);
      outputs_[index] = tensor;

      return tensor;
    }

    void *TestContext::Allocate(int64_t size, DataType dtype, DeviceType dev)
    {
      if (size == 0)
        size = 1;

      void *data = nullptr;
      if (dev == DeviceType::CPU_DEVICE)
      {
        data = host_allocator_.Allocate(size);
      }
      else if (dev == DeviceType::CUDA_DEVICE)
      {
        if (!memory_check_mode_)
        {
          data = cuda_allocator_.Allocate(size);
        }
        else
        {
          cudaMallocAsync(&data, size, stream_);
        }
      }
      else if (dev == DeviceType::CUDA_HOST_DEVICE)
      {
        data = pinned_allocator_.Allocate(size);
      }

      if (nullptr == data)
      {
        std::string err =
            "allocate memory fail, dev = " + std::to_string((int)(dev)) +
            "size = " + std::to_string(size);
        throw std::runtime_error(err);
      }

      return data;
    }

    bool TestContext::DeAllocate(void *ptr) { return true; }

    TensorPointer TestContext::Slice(TensorPointer tensor, int start, int size)
    {
      uint32_t row_size = 1;
      for (int i = 1; i < tensor->shape.size(); ++i)
      {
        row_size *= tensor->shape[i];
      }

      uint8_t *data = static_cast<uint8_t *>(tensor->raw_data);
      uint32_t element_size = SizeOfDataType(tensor->data_type);

      auto new_tensor = std::make_shared<ContextTensor>();
      new_tensor->raw_data = data + start * row_size * element_size;
      new_tensor->shape = tensor->shape;
      new_tensor->shape[0] = size;
      new_tensor->data_type = tensor->data_type;

      tensor->raw_data = new_tensor->raw_data;
      tensor->shape = new_tensor->shape;
      tensor->data_type = new_tensor->data_type;

      return new_tensor;
    }

    TensorPointer TestContext::SetOutput(int output_idx, int input_idx)
    {
      auto it = inputs_.find(input_idx);
      if (it == inputs_.end())
      {
        std::string err = std::string("SetOutput input_idx = ") +
                          std::to_string(input_idx) + "out of range";
        throw std::runtime_error(err);
      }

      TensorPointer tensor = it->second;
      outputs_[output_idx] = tensor;
      return tensor;
    }

    void *TestContext::GetUD() { return nullptr; }

    size_t TestContext::GetDeviceSharedMemPerBlock()
    {
      int size = 0;
      int device = 0;

      cudaGetDevice(&device);
      cudaDeviceGetAttribute(&size, cudaDevAttrMaxSharedMemoryPerBlock, device);
      return size;
    }

    void TestContext::SetInput(int index, TensorPointer tensor)
    {
      inputs_[index] = tensor;
    }

    void TestContext::SetOutputType(int index, DataType type)
    {
      output_data_types_[index] = type;
    }

    void TestContext::SetOutputDevice(int index, DeviceType type)
    {
      output_device_types_[index] = type;
    }

    TensorPointer TestContext::GetOutput(int index, DataType *data_type,
                                         DeviceType *device_type)
    {
      if (data_type)
      {
        auto it_data = output_data_types_.find(index);
        if (it_data != output_data_types_.end())
        {
          *data_type = it_data->second;
        }
      }

      if (device_type)
      {
        *device_type = DeviceType::CUDA_DEVICE;
        auto it_device = output_device_types_.find(index);
        if (it_device != output_device_types_.end())
        {
          *device_type = it_device->second;
        }
      }

      TensorPointer ret;
      auto it_tensor = outputs_.find(index);
      if (it_tensor != outputs_.end())
      {
        ret = it_tensor->second;
      }

      return ret;
    }

    TensorPointer TestContext::ToHostTensor(TensorPointer tensor)
    {
      uint64_t size = SizeOfDataType(tensor->data_type);
      auto shape = tensor->shape;
      for (const auto &x : shape)
      {
        size *= x;
      }

      void *data = nullptr;
      data = host_allocator_.Allocate(size);

      if (tensor->device_type == DeviceType::CUDA_DEVICE)
      {
        cudaMemcpyAsync(data, tensor->raw_data, size, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
      }
      else if (tensor->device_type == DeviceType::CPU_DEVICE)
      {
        memcpy(data, tensor->raw_data, size);
      }
      else if (tensor->device_type == DeviceType::CUDA_HOST_DEVICE)
      {
        memcpy(data, tensor->raw_data, size);
      }

      auto host_tensor = std::make_shared<ContextTensor>();
      host_tensor->raw_data = data;
      host_tensor->data_type = tensor->data_type;
      host_tensor->shape = shape;
      host_tensor->device_type = DeviceType::CPU_DEVICE;

      return host_tensor;
    }

    void TestContext::Stats()
    {
      if (!memory_check_mode_)
      {
        cuda_allocator_.Stats();
      }
      host_allocator_.Stats();
      pinned_allocator_.Stats();
    }

    void TestContext::Reset()
    {
      if (!memory_check_mode_)
      {
        cuda_allocator_.Reset();
      }
      host_allocator_.Reset();
      pinned_allocator_.Reset();
    }

    void FreelistAllocator::Manage(void *data, size_t size, const char *annotation,
                                   bool with_guard, cudaStream_t stream)
    {
      with_guard_ = with_guard;
      stream_ = stream;

      raw_base_ = static_cast<uint8_t *>(data);
      raw_size_ = size;

      uintptr_t base = (uintptr_t)(raw_base_);
      base = (base + align_bytes - 1) / align_bytes * align_bytes;

      base_ = (uint8_t *)(base);
      size_ = raw_base_ + size - base_;

      free_.push_back({0, size_});

      annotation_ = annotation;

      SetGuard();
    }

    void *FreelistAllocator::Allocate(size_t size)
    {
      void *ptr = nullptr;
      if (with_guard_)
      {
        size_t origin_size = size;
        size =
            (size + prefix_guard_length + suffix_guard_length + align_bytes - 1) /
            align_bytes * align_bytes;

        for (auto it = free_.begin(); it != free_.end(); ++it)
        {
          auto block_size = it->end - it->begin;
          if (block_size > size)
          {
            ptr = base_ + it->begin + prefix_guard_length;
            alloc_.push_back({it->begin, it->begin + size,
                              it->begin + prefix_guard_length,
                              it->begin + prefix_guard_length + origin_size});
            it->begin += size;
            break;
          }
          else if (block_size == size)
          {
            ptr = base_ + it->begin + prefix_guard_length;
            alloc_.push_back({it->begin, it->begin + size,
                              it->begin + prefix_guard_length,
                              it->begin + prefix_guard_length + origin_size});
            free_.erase(it);
            break;
          }
        }
      }
      else
      {
        size = (size + align_bytes - 1) / align_bytes * align_bytes;

        for (auto it = free_.begin(); it != free_.end(); ++it)
        {
          auto block_size = it->end - it->begin;
          if (block_size > size)
          {
            ptr = base_ + it->begin;
            alloc_.push_back({it->begin, it->begin + size});
            it->begin += size;
            break;
          }
          else if (block_size == size)
          {
            ptr = base_ + it->begin;
            alloc_.push_back({it->begin, it->begin + size});
            free_.erase(it);
            break;
          }
        }
      }

      return ptr;
    }

    void FreelistAllocator::Free(void *ptr)
    {
      // in our use case, we have no place to free memory,
      // so we implement it an exception.

      throw std::runtime_error("Free is not implemented yet");
    }

    void FreelistAllocator::Stats()
    {
      int i = 0;
      uint64_t total = 0;
      printf("---- TestContext Memory Usage Stats for [%.10s]\n",
             annotation_.c_str());
      printf(
          "- chunk -- size(B) ---- offset ----- begin-addr ------ end-addr ----\n");
      for (auto it = alloc_.begin(); it != alloc_.end(); ++it)
      {
        printf("| %04d | %10zu | %10lu | %p | %p |\n", i, it->end - it->begin,
               it->begin, base_ + it->begin, base_ + it->end);
        total += it->end - it->begin;
        ++i;
      }
      printf(
          "--------------------------------------------------------------------\n");
      printf("Total-Size: %-10zu                                             |\n",
             total);
      printf(
          "====================================================================\n");
    }

    void FreelistAllocator::Reset()
    {
      free_.clear();
      alloc_.clear();

      Manage(raw_base_, raw_size_, annotation_.c_str(), with_guard_, stream_);
    }

    __global__ void set_guard(uint8_t *data, int64_t size)
    {
      int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;

      if (idx < size)
      {
        data[idx] = 170; // bit: 10101010
      }
    }

    __global__ void guard_check(uint8_t *data, int64_t begin, int64_t end,
                                int64_t head, int64_t tail)
    {
      int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;

      if (idx >= 0 && idx < head)
      {
        if (data[idx] != 170)
        {
          printf("\033[1;31m==== CUDA MEMORY GUARD CHECK ERROR ======\n");
          printf(
              "prefix guard check fail size = %ld, touched = %ld, data = %p, begin "
              "= %ld, end = %ld, head = %ld, tail = %ld\n",
              tail - head, idx - head, data, begin, end, head, tail);
        }
      }

      if (idx >= tail && idx < end)
      {
        if (data[idx] != 170)
        {
          printf("\033[1;31m==== CUDA MEMORY GUARD CHECK ERROR ======\n");
          printf(
              "suffix guard check fail size = %ld, touched = %ld, data = %p, begin "
              "= %ld, end = %ld, head = %ld, tail = %ld\n",
              tail - head, idx - head, data, begin, end, head, tail);
        }
      }
    }

    void FreelistAllocator::SetGuard()
    {
      if (!with_guard_)
      {
        return;
      }

      dim3 block(512);
      dim3 grid((raw_size_ + block.x - 1) / block.x);

      set_guard<<<grid, block, 0, stream_>>>(raw_base_, raw_size_);
      cudaStreamSynchronize(stream_);
      auto err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        fprintf(stderr, "set guard fail: %s\n", cudaGetErrorString(err));
      }
    }

    void FreelistAllocator::CheckGuard()
    {
      if (!with_guard_)
      {
        return;
      }

      for (auto it = alloc_.begin(); it != alloc_.end(); ++it)
      {
        uint8_t *data = base_ + it->begin;
        int64_t begin = 0;
        int64_t end = it->end - it->begin;
        int64_t head = it->head - it->begin;
        int64_t tail = it->tail - it->begin;

        dim3 block(512);
        dim3 grid((tail - begin + block.x - 1) / block.x);

        guard_check<<<grid, block, 0, stream_>>>(data, begin, end, head, tail);
        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        if (err != cudaSuccess)
        {
          fprintf(stderr, "guard check fail: %s\n", cudaGetErrorString(err));
        }
      }
    }

  } // namespace test
} // namespace hpc_nn_ops
