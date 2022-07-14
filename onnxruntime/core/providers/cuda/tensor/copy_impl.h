// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/binary_elementwise_args.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void StridedCopyImpl(cudaStream_t stream, const T* src_data, T* dst_data, const BinaryElementwiseArgs& args);

}  // namespace cuda
}  // namespace onnxruntime
