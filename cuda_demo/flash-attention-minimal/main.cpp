#include <torch/extension.h>

torch::Tensor flash_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", torch::wrap_pybind_function(flash_attention_forward), "flash_attention_forward");
}