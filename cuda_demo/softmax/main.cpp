#include <torch/extension.h>

torch::Tensor softmax_2d_naive(torch::Tensor X, const int REPEAT_TIME=1);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_2d_naive", torch::wrap_pybind_function(softmax_2d_naive),
        "softmax_2d_naive");

}