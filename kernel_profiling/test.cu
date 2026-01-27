#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>

__global__ void dummy_kernel() { }

int main() {
    nvtxRangePushA("Test_NVTX");
    cudaProfilerStart();
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaProfilerStop();
    nvtxRangePop();
    return 0;
}