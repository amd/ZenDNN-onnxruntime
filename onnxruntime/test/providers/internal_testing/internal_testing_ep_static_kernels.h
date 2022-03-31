// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace internal_testing_ep {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* /*context*/) const override {
    ORT_NOT_IMPLEMENTED("Internal testing EP kernels are not expected to be executed.");
  }
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
