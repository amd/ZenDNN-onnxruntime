// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"


namespace onnxruntime {
struct CUDAExecutionProviderInfo;
struct OrtCUDAProviderOptions;
struct OrtCUDAProviderOptionsV2;

// defined in provider_bridge_ort.cc
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(
    const OrtCUDAProviderOptions* provider_options);

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(
    const OrtCUDAProviderOptionsV2* provider_options);

}  // namespace onnxruntime
