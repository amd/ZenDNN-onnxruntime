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

#include "lib/Api.Ort/pch.h"
#include "OnnxruntimeCpuSessionBuilder.h"
#include "OnnxruntimeEngine.h"
#include "OnnxruntimeErrors.h"

using namespace _winml;

HRESULT OnnxruntimeCpuSessionBuilder::RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory) {
  engine_factory_ = engine_factory;
  return S_OK;
}

HRESULT
OnnxruntimeCpuSessionBuilder::CreateSessionOptions(
    OrtSessionOptions** options) {
  RETURN_HR_IF_NULL(E_POINTER, options);

  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtSessionOptions* ort_options;
  RETURN_HR_IF_NOT_OK_MSG(ort_api->CreateSessionOptions(&ort_options),
                          ort_api);

  auto session_options = UniqueOrtSessionOptions(ort_options, ort_api->ReleaseSessionOptions);

  // set the graph optimization level to all (used to be called level 3)
  RETURN_HR_IF_NOT_OK_MSG(ort_api->SetSessionGraphOptimizationLevel(session_options.get(), GraphOptimizationLevel::ORT_ENABLE_ALL),
                          ort_api);

#ifndef _WIN64
  auto use_arena = false;
#else
  auto use_arena = true;
#endif
  // RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_CPU(session_options.get(), use_arena),
  //                         ort_api);
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_Zendnn(session_options.get(), use_arena),
                           ort_api);

  // call release() so the underlying OrtSessionOptions object isn't freed
  *options = session_options.release();

  return S_OK;
}

HRESULT
OnnxruntimeCpuSessionBuilder::CreateSession(
    OrtSessionOptions* options,
    OrtThreadPool* inter_op_thread_pool,
    OrtThreadPool* intra_op_thread_pool,
    OrtSession** session) {
  RETURN_HR_IF_NULL(E_POINTER, session);

  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtEnv* ort_env;
  RETURN_IF_FAILED(engine_factory_->GetOrtEnvironment(&ort_env));

  OrtSession* ort_session_raw;
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->CreateSessionWithoutModel(ort_env, options, inter_op_thread_pool, intra_op_thread_pool, &ort_session_raw),
                          engine_factory_->UseOrtApi());

  auto ort_session = UniqueOrtSession(ort_session_raw, ort_api->ReleaseSession);

  *session = ort_session.release();

  return S_OK;
}

HRESULT
OnnxruntimeCpuSessionBuilder::Initialize(
    OrtSession* session) {
  RETURN_HR_IF_NULL(E_INVALIDARG, session);

  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->SessionInitialize(session),
                          engine_factory_->UseOrtApi());

  return S_OK;
}
