#include "embedding.cuh"

// const int REPEAT_TIME = 1000;
const int BLOCK_X = 128;
__global__ void embedding_naive_kernel(const int* input, float* weight, float* output, const int n, const int emb_size){
    /**
     * @brief 
     * input 可以看成是一个1D的输入tensor[N]
     * 然后output可以看成是一个[N,EMB]tensor
     * weight就应该是[VecSize, EMB]的tensor
     * 这种就相当于是GM2GM之间的vector写？
     */
    // 一个thread处理一个输出，很合理
    // 如果说EMB不大，可以在block里面搞一个smem作为类似hashmap的存在，这样可以减少一部分GM的读
    int tid = blockIdx.x;
    // if(blockIdx.x >= n) return;
    float* output_now = output + tid * emb_size;
    const int index = input[tid]; 
    float* weight_now = weight + index * emb_size; 

    for (size_t i = threadIdx.x; i < emb_size; i+=blockDim.x)
    {
        // 这里可以搞成float4？
        output_now[i] = weight_now[i];
    }
    // memcpy(output_now, weight_now, emb_size * sizeof(float));
}



torch::Tensor embedding_naive(torch::Tensor X, torch::Tensor Weight, const int REPEAT_TIME=1) {
  std::cout << "embedding_naive" << std::endl;
  const int M = X.size(0);
  const int emb = Weight.size(1);
  std::cout<< "emb:" << emb << std::endl;
  std::cout<< "M:" << M << std::endl;
  auto Y = torch::zeros({M, emb}, torch::kCUDA);
  dim3 block_dim(128);
  dim3 grid_dim(6);
  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < REPEAT_TIME; ++i)
    embedding_naive_kernel<<<grid_dim, block_dim>>>(
        X.data_ptr<int>(), Weight.data_ptr<float>(), Y.data_ptr<float>(), M, emb);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "embedding_naive Kernel execution time: "
            << milliseconds / REPEAT_TIME << " ms" << std::endl;
  return Y;
}
