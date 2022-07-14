// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/complex_mul_impl.h"

#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, bool IsConj>
struct OP_ComplexMul {
  using Complex = aligned_vector<T, 2>;
  __device__ __inline__ Complex operator()(Complex a, Complex b) const {
    Complex result;
    if (IsConj) {
      result.val[0] = a.val[0] * b.val[0] + a.val[1] * b.val[1];
      result.val[1] = a.val[1] * b.val[0] - a.val[0] * b.val[1];
    } else {
      result.val[0] = a.val[0] * b.val[0] - a.val[1] * b.val[1];
      result.val[1] = a.val[0] * b.val[1] + a.val[1] * b.val[0];
    }
    return result;
  }
};

template <typename T, bool IsConj>
void ComplexMul_Impl(cudaStream_t stream, const T* lhs_data, const T* rhs_data, T* output_data,
                     const BinaryElementwiseArgs& args) {
  using Complex = aligned_vector<T, 2>;
  constexpr int complex_alignment = std::alignment_of<Complex>::value;
  ORT_ENFORCE(reinterpret_cast<uint64_t>(lhs_data) % complex_alignment == 0 &&
              reinterpret_cast<uint64_t>(rhs_data) % complex_alignment == 0 &&
              reinterpret_cast<uint64_t>(output_data) % complex_alignment == 0);
  const Complex* lhs_complex_data = reinterpret_cast<const Complex*>(lhs_data);
  const Complex* rhs_complex_data = reinterpret_cast<const Complex*>(rhs_data);
  Complex* output_complex_data = reinterpret_cast<Complex*>(output_data);
  OP_ComplexMul<T, IsConj> op;
  BinaryElementWiseImpl<Complex, Complex, Complex, decltype(op)>(stream, lhs_complex_data, rhs_complex_data,
                                                                 output_complex_data, args, op);
};

#define SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(T, IsConj)                                                                  \
  template void ComplexMul_Impl<T, IsConj>(cudaStream_t stream, const T* lhs_data, const T* rhs_data, T* output_data, \
                                           const BinaryElementwiseArgs& args);

SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(float, true)
SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(half, true)
SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(float, false)
SPECIALIZE_STACKEDCOMPLEXMUL_IMPL(half, false)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
