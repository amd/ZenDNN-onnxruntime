/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*******************************************************************************
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*******************************************************************************/

#include "core/providers/get_execution_providers.h"

#include "core/graph/constants.h"
#include "core/common/common.h"

#include <string_view>

namespace onnxruntime {

namespace {
struct ProviderInfo {
  std::string_view name;
  bool available;
};

// all providers ordered by default priority from highest to lowest
// kCpuExecutionProvider should always be last
constexpr ProviderInfo kProvidersInPriorityOrder[] =
    {
        {
            kTensorrtExecutionProvider,
#ifdef USE_TENSORRT
            true,
#else
            false,
#endif
        },
        {
            kCudaExecutionProvider,
#ifdef USE_CUDA
            true,
#else
            false,
#endif
        },
        {
            kMIGraphXExecutionProvider,
#ifdef USE_MIGRAPHX
            true,
#else
            false,
#endif
        },
        {
            kRocmExecutionProvider,
#ifdef USE_ROCM
            true,
#else
            false,
#endif
        },
        {
            kOpenVINOExecutionProvider,
#ifdef USE_OPENVINO
            true,
#else
            false,
#endif
        },
        {
            kDnnlExecutionProvider,
#ifdef USE_DNNL
            true,
#else
            false,
#endif
        },
        {
            kZendnnExecutionProvider,
#ifdef USE_ZENDNN
            true,
#else
            false,
#endif
        },
        {
            kTvmExecutionProvider,
#ifdef USE_TVM
            true,
#else
            false,
#endif
        },
        {
            kVitisAIExecutionProvider,
#ifdef USE_VITISAI
            true,
#else
            false,
#endif
        },
        {
            kQnnExecutionProvider,
#ifdef USE_QNN
            true,
#else
            false,
#endif
        },
        {
            kNnapiExecutionProvider,
#ifdef USE_NNAPI
            true,
#else
            false,
#endif
        },
        {
            kJsExecutionProvider,
#ifdef USE_JSEP
            true,
#else
            false,
#endif
        },
        {
            kCoreMLExecutionProvider,
#ifdef USE_COREML
            true,
#else
            false,
#endif
        },
        {
            kArmNNExecutionProvider,
#ifdef USE_ARMNN
            true,
#else
            false,
#endif
        },
        {
            kAclExecutionProvider,
#ifdef USE_ACL
            true,
#else
            false,
#endif
        },
        {
            kDmlExecutionProvider,
#ifdef USE_DML
            true,
#else
            false,
#endif
        },
        {
            kRknpuExecutionProvider,
#ifdef USE_RKNPU
            true,
#else
            false,
#endif
        },
        {
            kXnnpackExecutionProvider,
#ifdef USE_XNNPACK
            true,
#else
            false,
#endif
        },
        {
            kCannExecutionProvider,
#ifdef USE_CANN
            true,
#else
            false,
#endif
        },
        {
            kAzureExecutionProvider,
#ifdef USE_AZURE
            true,
#else
            false,
#endif
        },
        {kCpuExecutionProvider, true},  // kCpuExecutionProvider is always last
};

constexpr size_t kAllExecutionProvidersCount = sizeof(kProvidersInPriorityOrder) / sizeof(ProviderInfo);

}  // namespace

const std::vector<std::string>& GetAllExecutionProviderNames() {
  static const auto all_execution_providers = []() {
    std::vector<std::string> result{};
    result.reserve(kAllExecutionProvidersCount);
    for (const auto& provider : kProvidersInPriorityOrder) {
      ORT_ENFORCE(provider.name.size() <= kMaxExecutionProviderNameLen, "Make the EP:", provider.name, " name shorter");
      result.push_back(std::string(provider.name));
    }
    return result;
  }();

  return all_execution_providers;
}

const std::vector<std::string>& GetAvailableExecutionProviderNames() {
  static const auto available_execution_providers = []() {
    std::vector<std::string> result{};
    for (const auto& provider : kProvidersInPriorityOrder) {
      ORT_ENFORCE(provider.name.size() <= kMaxExecutionProviderNameLen, "Make the EP:", provider.name, " name shorter");
      if (provider.available) {
        result.push_back(std::string(provider.name));
      }
    }
    return result;
  }();

  return available_execution_providers;
}

}  // namespace onnxruntime
