// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
namespace tvm{
  struct TvmEPOptions;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const tvm::TvmEPOptions& info);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const char* params);

}  // namespace onnxruntime
