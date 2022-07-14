// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

class Tensor;

namespace common {
class Status;
}  // namespace common

namespace cuda {

common::Status StridedCopyTensor(cudaStream_t stream, const Tensor& src, Tensor& dst);

}  // namespace cuda
}  // namespace onnxruntime
