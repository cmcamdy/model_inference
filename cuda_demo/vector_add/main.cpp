#include <torch/extension.h>

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", torch::wrap_pybind_function(vector_add), "vector_add");
}
