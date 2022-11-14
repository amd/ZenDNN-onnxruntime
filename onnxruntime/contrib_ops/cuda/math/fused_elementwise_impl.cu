// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/fused_elementwise_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl_functors.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, bool IsScalar, typename FuncT>
struct Operand {
  Operand(const T* data, FuncT func) : data_(data), func_(func) {}

  __device__ __forceinline__ T operator()(int linear_idx, T a) const {
    return func_(a, data_[IsScalar ? 0 : linear_idx]);
  }

  __device__ __forceinline__ T get(int linear_idx) const { return data_[IsScalar ? 0 : linear_idx]; }

  const T* data_;
  FuncT func_;
};

template <typename T, typename OperandT>
__device__ T ApplyOperands(int linear_idx, OperandT operand) {
  return operand.get(linear_idx);
};

template <typename T, typename OperandT, typename... OperandsT>
__device__ T ApplyOperands(int linear_idx, OperandT last_operand, OperandsT... other_operands) {
  return last_operand(linear_idx, ApplyOperands<T, OperandsT...>(linear_idx, other_operands...));
};

template <typename T, typename... OperandsT>
__global__ void FusedElementwiseKernel(T* output, const int N, OperandsT... operands) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = ApplyOperands<T, OperandsT...>(id, operands...);
}

template <typename T>
void FusedElementwiseImpl(cudaStream_t stream, int num_ops, T* output, const T* input1, const T* input2,
                          const T* input3, const T* input4, bool is_input1_scalar, bool is_input2_scalar,
                          bool is_input3_scalar, bool is_input4_scalar, OpType op1_type, OpType op2_type,
                          OpType op3_type, int N) {
  if (N == 0) return;

  using FuncAddType = OP_Add<T, T, T>;
  using FuncSubType = OP_Sub<T, T, T>;
  using FuncMulType = OP_Mul<T, T, T>;
  using FuncDivType = OP_Div<T, T, T>;

#define CASE_OP3_TYPE(is_input4_scalar_value, op3_type_value, func3_type)                                              \
  case op3_type_value: {                                                                                               \
    Operand<T, is_input4_scalar_value, func3_type> operand4(input4, func3_type());                                     \
    FusedElementwiseKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(output, N, operand4, operand3, \
                                                                                        operand2, operand1);           \
  } break

#define HANDLE_OP3_TYPE(is_input4_scalar_value)                      \
  switch (op3_type) {                                                \
    CASE_OP3_TYPE(is_input4_scalar_value, OpType::Add, FuncAddType); \
    CASE_OP3_TYPE(is_input4_scalar_value, OpType::Sub, FuncSubType); \
    CASE_OP3_TYPE(is_input4_scalar_value, OpType::Mul, FuncMulType); \
    CASE_OP3_TYPE(is_input4_scalar_value, OpType::Div, FuncDivType); \
  }

#define CASE_OP2_TYPE(is_input3_scalar_value, op2_type_value, func2_type)                                      \
  case op2_type_value: {                                                                                       \
    Operand<T, is_input3_scalar_value, func2_type> operand3(input3, func2_type());                             \
    if (num_ops == 2) {                                                                                        \
      FusedElementwiseKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(output, N, operand3, \
                                                                                          operand2, operand1); \
      return;                                                                                                  \
    }                                                                                                          \
    if (is_input4_scalar) {                                                                                    \
      HANDLE_OP3_TYPE(true);                                                                                   \
    } else {                                                                                                   \
      HANDLE_OP3_TYPE(false);                                                                                  \
    }                                                                                                          \
  } break

#define HANDLE_OP2_TYPE(is_input3_scalar_value)                      \
  switch (op2_type) {                                                \
    CASE_OP2_TYPE(is_input3_scalar_value, OpType::Add, FuncAddType); \
    CASE_OP2_TYPE(is_input3_scalar_value, OpType::Sub, FuncSubType); \
    CASE_OP2_TYPE(is_input3_scalar_value, OpType::Mul, FuncMulType); \
    CASE_OP2_TYPE(is_input3_scalar_value, OpType::Div, FuncDivType); \
  }

#define CASE_OP1_TYPE(is_input2_scalar_value, op1_type_value, func1_type)          \
  case op1_type_value: {                                                           \
    Operand<T, is_input2_scalar_value, func1_type> operand2(input2, func1_type()); \
    if (is_input3_scalar) {                                                        \
      HANDLE_OP2_TYPE(true);                                                       \
    } else {                                                                       \
      HANDLE_OP2_TYPE(false);                                                      \
    }                                                                              \
  } break

#define HANDLE_OP1_TYPE(is_input2_scalar_value)                      \
  switch (op1_type) {                                                \
    CASE_OP1_TYPE(is_input2_scalar_value, OpType::Add, FuncAddType); \
    CASE_OP1_TYPE(is_input2_scalar_value, OpType::Sub, FuncSubType); \
    CASE_OP1_TYPE(is_input2_scalar_value, OpType::Mul, FuncMulType); \
    CASE_OP1_TYPE(is_input2_scalar_value, OpType::Div, FuncDivType); \
  }

#define HANDLE_INPUT2_SCALAR_TYPE(is_input1_scalar_value)                                  \
  Operand<T, is_input1_scalar_value, OP_Add<T, T, T>> operand1(input1, OP_Add<T, T, T>()); \
  if (is_input2_scalar) {                                                                  \
    HANDLE_OP1_TYPE(true);                                                                 \
  } else {                                                                                 \
    HANDLE_OP1_TYPE(false);                                                                \
  }

  ORT_ENFORCE(num_ops == 2 || num_ops == 3);
  int blocks_per_grid = CeilDiv(N, GridDim::maxThreadsPerBlock);
  if (is_input1_scalar) {
    HANDLE_INPUT2_SCALAR_TYPE(true);
  } else {
    HANDLE_INPUT2_SCALAR_TYPE(false);
  }
}

#define SPECIALIZED_FUSED_ELEMENTWISE_IMPL(T)                                                                          \
  template void FusedElementwiseImpl<T>(cudaStream_t stream, int num_ops, T* output, const T* input1, const T* input2, \
                                        const T* input3, const T* input4, bool is_input1_scalar,                       \
                                        bool is_input2_scalar, bool is_input3_scalar, bool is_input4_scalar,           \
                                        OpType op1_type, OpType op2_type, OpType op3_type, int N);

SPECIALIZED_FUSED_ELEMENTWISE_IMPL(float)
SPECIALIZED_FUSED_ELEMENTWISE_IMPL(half)

#undef SPECIALIZED_FUSED_ELEMENTWISE_IMPL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
