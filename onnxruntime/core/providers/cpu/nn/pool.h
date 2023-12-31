// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {

template <typename T, typename PoolType>
class Pool : public OpKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
    const std::string& op_name = info.GetKernelDef().OpName();
    if (op_name == "LpPool" || op_name == "GlobalLpPool") {
      pool_context_.init(info);
    }
  }

  ~Pool() override = default;

  Status Compute(OpKernelContext* context) const override;

 private:
  PoolProcessContext pool_context_;
};

// For averagepool v19 and beyond
// version 19: Added dilations
template <typename T>
class AveragePoolV19 : public OpKernel, public PoolBase {
 public:
  AveragePoolV19(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t p_;
};

// For maxpool v8 and beyond
// version 8: Added storage_order And Indices
// version 10: Added ceil_mode
// version 11: Added dilations
// version 12: Added int8/uint8 support
class MaxPoolV8 : public OpKernel, public PoolBase {
  template <typename T>
  struct ComputeHelper {
    Status operator()(const MaxPoolV8* inst, OpKernelContext* context) const {
      return inst->ComputeImpl<T>(context);
    }
  };

 public:
  MaxPoolV8(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {}
  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;
};

// For lppool v18 and beyond
// version 18: Added ceil_mode and dilations
template <typename T>
class LpPoolV18 : public OpKernel, public PoolBase {
 public:
  LpPoolV18(const OpKernelInfo& info) : OpKernel(info), PoolBase(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("p", &p_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t p_;
};

}  // namespace onnxruntime
