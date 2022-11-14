// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum class OpType : int32_t {
  Add = (int32_t)0,
  Sub = (int32_t)1,
  Mul = (int32_t)2,
  Div = (int32_t)3,
};

template <typename T>
void FusedElementwiseImpl(cudaStream_t stream, int num_ops, T* output, const T* input1, const T* input2,
                          const T* input3, const T* input4, bool is_input1_scalar, bool is_input2_scalar,
                          bool is_input3_scalar, bool is_input4_scalar, OpType op1_type, OpType op2_type,
                          OpType op3_type, int N);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
