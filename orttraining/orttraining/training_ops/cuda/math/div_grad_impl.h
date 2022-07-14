// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/binary_elementwise_args.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void ImplDivGrad(cudaStream_t stream, const T* a_data, const T* b_data, const T* dy_data, T* da_output_data,
                 T* db_output_data, const BinaryElementwiseArgs& args);

}  // namespace cuda
}  // namespace onnxruntime
