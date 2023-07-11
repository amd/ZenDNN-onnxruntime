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

#include "adapter/pch.h"

#include "winml_adapter_c_api.h"
#include "winml_adapter_apis.h"
#include "core/session/ort_apis.h"

#include <core/providers/winml/winml_provider_factory.h>
#include <core/providers/cpu/cpu_provider_factory.h>
#include <core/providers/zendnn/zendnn_provider_factory.h>

const OrtApi* GetVersion1Api();

namespace winmla = Windows::AI::MachineLearning::Adapter;

static constexpr WinmlAdapterApi winml_adapter_api_1 = {
    // Schema override
    &winmla::OverrideSchema,

    // OrtEnv methods
    &winmla::EnvConfigureCustomLoggerAndProfiler,

    // OrtModel methods
    &winmla::CreateModelFromPath,
    &winmla::CreateModelFromData,
    &winmla::CloneModel,
    &winmla::ModelGetAuthor,
    &winmla::ModelGetName,
    &winmla::ModelSetName,
    &winmla::ModelGetDomain,
    &winmla::ModelGetDescription,
    &winmla::ModelGetVersion,
    &winmla::ModelGetInputCount,
    &winmla::ModelGetOutputCount,
    &winmla::ModelGetInputName,
    &winmla::ModelGetOutputName,
    &winmla::ModelGetInputDescription,
    &winmla::ModelGetOutputDescription,
    &winmla::ModelGetInputTypeInfo,
    &winmla::ModelGetOutputTypeInfo,
    &winmla::ModelGetMetadataCount,
    &winmla::ModelGetMetadata,
    &winmla::ModelEnsureNoFloat16,
    &winmla::SaveModel,

    // OrtSessionOptions methods
    &OrtSessionOptionsAppendExecutionProvider_CPU,
    &OrtSessionOptionsAppendExecutionProvider_Zendnn,
    &winmla::OrtSessionOptionsAppendExecutionProviderEx_DML,

    // OrtSession methods
    &winmla::CreateSessionWithoutModel,
    &winmla::SessionGetExecutionProvider,
    &winmla::SessionInitialize,
    &winmla::SessionRegisterGraphTransformers,
    &winmla::SessionRegisterCustomRegistry,
    &winmla::SessionLoadAndPurloinModel,
    &winmla::SessionStartProfiling,
    &winmla::SessionEndProfiling,
    &winmla::SessionCopyOneInputAcrossDevices,
    &winmla::SessionGetNumberOfIntraOpThreads,
    &winmla::SessionGetIntraOpThreadSpinning,
    &winmla::SessionGetNamedDimensionsOverrides,

    // Dml methods (TODO need to figure out how these need to move to session somehow...)
    &winmla::DmlExecutionProviderSetDefaultRoundingMode,
    &winmla::DmlExecutionProviderFlushContext,
    &winmla::DmlExecutionProviderReleaseCompletedReferences,
    &winmla::DmlCopyTensor,

    &winmla::GetProviderMemoryInfo,
    &winmla::GetProviderAllocator,
    &winmla::FreeProviderAllocator,

    &winmla::ExecutionProviderSync,

    &winmla::CreateCustomRegistry,

    &winmla::ValueGetDeviceId,
    &winmla::SessionGetInputRequiredDeviceId,

    &winmla::CreateTensorTypeInfo,
    &winmla::CreateSequenceTypeInfo,
    &winmla::CreateMapTypeInfo,
    &winmla::CreateModel,
    &winmla::ModelAddInput,
    &winmla::ModelAddConstantInput,
    &winmla::ModelAddOutput,
    &winmla::ModelAddOperator,
    &winmla::ModelGetOpsetVersion,
    &winmla::OperatorGetNumInputs,
    &winmla::OperatorGetInputName,
    &winmla::OperatorGetNumOutputs,
    &winmla::OperatorGetOutputName,
    &winmla::JoinModels,
    &winmla::CreateThreadPool,

    // Release
    &winmla::ReleaseModel,
    &winmla::ReleaseThreadPool,
};

const WinmlAdapterApi* ORT_API_CALL OrtGetWinMLAdapter(_In_ uint32_t ort_version) NO_EXCEPTION {
  if (ort_version >= 2) {
    return &winml_adapter_api_1;
  }

  return nullptr;
}