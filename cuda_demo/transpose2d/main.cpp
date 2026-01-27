#include <torch/extension.h>

torch::Tensor transpose2d_naive(torch::Tensor X);
torch::Tensor transpose2d_vector(torch::Tensor X);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose2d_naive", torch::wrap_pybind_function(transpose2d_naive),
        "transpose2d_naive");
  m.def("transpose2d_vector", torch::wrap_pybind_function(transpose2d_vector),
        "transpose2d_vector");
}