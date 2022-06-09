// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct OrtTensorRTProviderOptions;
struct OrtTensorRTProviderOptionsV2;

// defined in provider_bridge_ort.cc
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(int device_id);

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(
    const OrtTensorRTProviderOptions* params);

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(
    const OrtTensorRTProviderOptionsV2* params);

}  // namespace onnxruntime
