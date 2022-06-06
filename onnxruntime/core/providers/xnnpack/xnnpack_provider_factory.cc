// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/xnnpack_provider_factory.h"

#include "core/framework/error_code_helper.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct XnnpackProviderFactory : IExecutionProviderFactory {
  XnnpackProviderFactory() = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> XnnpackProviderFactory::CreateProvider() {
  XnnpackExecutionProviderInfo info;
  return std::make_unique<XnnpackExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Xnnpack() {
  return std::make_shared<XnnpackProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Xnnpack,
                    _In_ OrtSessionOptions* options,
                    _In_ const OrtProviderOptions* /*provider_options*/) {
  API_IMPL_BEGIN
  // no provider options currently so ignore param. provider_options may be nullptr
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Xnnpack());
  return nullptr;
  API_IMPL_END
}
