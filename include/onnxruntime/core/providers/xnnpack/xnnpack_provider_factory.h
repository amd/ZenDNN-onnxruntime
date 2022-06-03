// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Xnnpack,
                          _In_ OrtSessionOptions* options,
                          _In_ const OrtProviderOptions* provider_options);  // provider_options may be nullptr

#ifdef __cplusplus
}
#endif
