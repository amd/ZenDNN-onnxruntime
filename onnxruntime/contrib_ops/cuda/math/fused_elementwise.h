// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class FusedElementwise final : public onnxruntime::cuda::CudaKernel {
 public:
  FusedElementwise(const OpKernelInfo& info) : CudaKernel{info} {
    ORT_ENFORCE(info.GetAttrs("op_types", op_types_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::vector<std::string> op_types_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
