#include <stdio.h>
#include <iostream>
#include <chrono>
// 一个示例kernel，之所以写result，是为了避免被nvcc优化掉
__global__ void kernel_test1(float *result)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 1234; i >= 0; --i)
    {
        if (idx == 0)
        {
            result[0] = i;
        }
    }
}

__global__ void kernel_test2(float *result)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = 1234 * 2; i >= 0; --i)
    {
        if (idx == 0)
        {
            result[1] = i;
        }
    }
}

int main()
{
    // 一些调用kernel的准备工作
    const dim3 block_size(5, 6, 7);
    const dim3 grid_size(2, 3, 4);
    float *d_result;
    cudaMalloc((void **)&d_result, 2 * sizeof(float));

    // 一个简单的CPU函数，CPU耗时统计示例
    auto t1 = std::chrono::high_resolution_clock::now();
    float localResult = 0.0f;
    for (unsigned int i = 0; i < 123456789; ++i)
    {
        localResult += float(i) * 0.00001f;
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // 这里预先调用几次kernel，预热下GPU，避免预热的影响
    for (int i = 0; i < 10; ++i)
    {
        kernel_test1<<<grid_size, block_size>>>(d_result);
        kernel_test2<<<grid_size, block_size>>>(d_result);
    }
    cudaDeviceSynchronize();

    // 这里实际上是统计的kernel launcher的时间
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
    {
        kernel_test1<<<grid_size, block_size>>>(d_result);
        kernel_test2<<<grid_size, block_size>>>(d_result);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    
    // 等待kernel执行完成
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();

    auto d21 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    auto d43 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
    auto d54 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4);

    std::cout << "CPU:" << d21.count() << " ms" << std::endl;
    std::cout << "Kernel Launcher:" << d43.count() << " ms," << std::endl;
    std::cout << "cudaDeviceSynchronize:" << d54.count() << " ms" << std::endl;
    
    cudaFree(d_result);
    return 0;
}