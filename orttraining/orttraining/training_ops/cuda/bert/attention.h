// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/framework/random_generator.h"

namespace onnxruntime {
namespace cuda {

class AttentionTraining final : public CudaKernel {
 public:
  explicit AttentionTraining(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

class AttentionTrainingGrad final : public CudaKernel {
 public:
  explicit AttentionTrainingGrad(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace cuda
}  // namespace onnxruntime
