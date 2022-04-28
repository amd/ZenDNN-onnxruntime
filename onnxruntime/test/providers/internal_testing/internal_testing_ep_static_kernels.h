// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace internal_testing_ep {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* /*context*/) const override;

  // use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  std::unique_ptr<Tensor> packed_w_;
};


class InvalidNchwKernel : public OpKernel {
 public:
  InvalidNchwKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* /*context*/) const override {
    ORT_THROW(
        "Layout transformation in GraphPartitioner should have replaced this node with one in the "
        "kMSInternalNHWCDomain domain.");
  }
};
}  // namespace internal_testing_ep
}  // namespace onnxruntime
