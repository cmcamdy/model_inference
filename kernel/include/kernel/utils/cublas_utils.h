#ifndef UTILS_CUBLAS_UTILS_H_
#define UTILS_CUBLAS_UTILS_H_

#include <cublas_v2.h>
#include "hpc_nn_ops/utils/error.h"

#include <cstdint>
#if CUDART_VERSION >= 10010
#include <cublasLt.h>
#endif  // CUDART_VERSION >= 10010

inline const char* GetCublasErrorString(int error) {
  switch (error) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unrecognized error";
}

#ifndef CHECK_CUBLAS_ERROR
#define CHECK_CUBLAS_ERROR(fn)                                               \
  do {                                                                       \
    int error = static_cast<int>(fn);                                        \
    CHECK_COND(error == CUBLAS_STATUS_SUCCESS, GetCublasErrorString(error)); \
  } while (0)  // ; intentionally left off.
#endif         // CHECK_CUBLAS_ERROR

namespace utils {
namespace cublas {

inline cublasOperation_t CUBLASBooleanToTranspose(bool item) {
  return item ? CUBLAS_OP_T : CUBLAS_OP_N;
}

template <typename DType>
void CublasGemm(bool trans_a, bool trans_b, uint64_t m, uint64_t n, uint64_t k, DType* matrix_A,
                DType* matrix_B, DType* matrix_C, float alpha = 1.0, float beta = 0.,
                cudaStream_t stream = 0);

}  // namespace cublas
}  // namespace utils

#endif  // !UTILS_CUBLAS_UTILS_H_
