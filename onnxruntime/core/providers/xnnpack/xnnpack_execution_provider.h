// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"

namespace onnxruntime {

// Information needed to construct Xnnpack execution providers. Stub for future use.
struct XnnpackExecutionProviderInfo {
  bool create_arena{true};

  explicit XnnpackExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  XnnpackExecutionProviderInfo() = default;
};

class XnnpackExecutionProvider : public IExecutionProvider {
 public:
  XnnpackExecutionProvider(const XnnpackExecutionProviderInfo& info);
  ~XnnpackExecutionProvider() override;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  DataLayout GetPreferredLayout() const override { return DataLayout::NHWC; }
};

}  // namespace onnxruntime
