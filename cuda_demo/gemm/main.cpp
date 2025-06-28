#include <torch/extension.h>

torch::Tensor smem_gemm(torch::Tensor a, torch::Tensor b);
torch::Tensor naive_gemm(torch::Tensor a, torch::Tensor b);
torch::Tensor smem_gemm_blocktiling_1d(torch::Tensor a, torch::Tensor b);
torch::Tensor smem_gemm_blocktiling_2d(torch::Tensor a, torch::Tensor b);
torch::Tensor smem_gemm_blocktiling_2d_vector(torch::Tensor a, torch::Tensor b);
torch::Tensor smem_gemm_blocktiling_2d_vector_avoid_bank_conflict(
    torch::Tensor a, torch::Tensor b);
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("smem_gemm_blocktiling_2d_vector_avoid_bank_conflict",
        torch::wrap_pybind_function(
            smem_gemm_blocktiling_2d_vector_avoid_bank_conflict),
        "smem_gemm_blocktiling_2d_vector_avoid_bank_conflict");
  m.def("smem_gemm_blocktiling_2d_vector",
        torch::wrap_pybind_function(smem_gemm_blocktiling_2d_vector),
        "smem_gemm_blocktiling_2d_vector");
  m.def("smem_gemm_blocktiling_2d",
        torch::wrap_pybind_function(smem_gemm_blocktiling_2d),
        "smem_gemm_blocktiling_2d");
  m.def("smem_gemm_blocktiling_1d",
        torch::wrap_pybind_function(smem_gemm_blocktiling_1d),
        "smem_gemm_blocktiling_1d");
  m.def("smem_gemm", torch::wrap_pybind_function(smem_gemm), "smem_gemm");
  m.def("naive_gemm", torch::wrap_pybind_function(naive_gemm), "naive_gemm");
}
