#include <torch/extension.h>


void rms_norm_f32(torch::Tensor x, torch::Tensor y, float g);
void rms_norm_f32x4(torch::Tensor x, torch::Tensor y, float g);
void rms_norm_f32x4v2(torch::Tensor x, torch::Tensor y, float g);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm_f32", torch::wrap_pybind_function(rms_norm_f32), "rms_norm_f32");
  m.def("rms_norm_f32x4", torch::wrap_pybind_function(rms_norm_f32x4), "rms_norm_f32x4");
  m.def("rms_norm_f32x4v2", torch::wrap_pybind_function(rms_norm_f32x4v2), "rms_norm_f32x4v2");
}