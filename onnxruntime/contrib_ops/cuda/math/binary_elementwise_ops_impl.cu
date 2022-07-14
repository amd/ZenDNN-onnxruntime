// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "contrib_ops/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define OP(name, expr)                                   \
  template <class T>                                     \
  struct OP_##name {                                     \
    __device__ __inline__ T operator()(T a, T b) const { \
      return (expr);                                     \
    }                                                    \
  };

#define CONTRIB_BINARY_ELEMENTWISE_IMPL(name)                                             \
  CONTRIB_BINARY_ELEMENTWISE_IMPL_DECLARATION(name) {                                     \
    BinaryElementWiseImpl(stream, lhs_data, rhs_data, output_data, args, OP_##name<T>()); \
  }

#define CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, T)                                              \
  template void Impl_##x<T>(cudaStream_t stream, const T* lhs_data, const T* rhs_data, T* output_data, \
                            const BinaryElementwiseArgs& args);

#define CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL_UZILHFD(x) \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, uint32_t)     \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, uint64_t)     \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int32_t)      \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int64_t)      \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)         \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)        \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)

#define CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL_OIL(x) \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, bool)     \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int32_t)  \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, int64_t)

#define CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(x) \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, half)     \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, float)    \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, double)   \
  CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL(x, BFloat16)

// create declarations for op and impl
#define CONTRIB_BINARY_OP_NAME_EXPR(name, expr) \
  OP(name, expr)                                \
  CONTRIB_BINARY_ELEMENTWISE_IMPL(name)

CONTRIB_BINARY_OPS()

#undef CONTRIB_BINARY_OP_NAME_EXPR

// create specialized impl
// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

CONTRIB_SPECIALIZED_BINARY_ELEMENTWISE_IMPL_HFD(BiasGelu)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
