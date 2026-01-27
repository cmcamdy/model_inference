#include <torch/extension.h>

torch::Tensor embedding_naive(torch::Tensor X, torch::Tensor Weight, const int REPEAT_TIME=1);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("embedding_naive", torch::wrap_pybind_function(embedding_naive),
        "embedding_naive");
}