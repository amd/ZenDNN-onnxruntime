// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/provider_options.h"
#include "core/providers/providers.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory>
CreateExecutionProviderFactory_SNPE(const ProviderOptions& provider_options_map);

}  // namespace onnxruntime
