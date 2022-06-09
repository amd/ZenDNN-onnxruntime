// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct OrtOpenVINOProviderOptions;

// defined in provider_bridge_ort.cc
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const OrtOpenVINOProviderOptions* provider_options);

}  // namespace onnxruntime
