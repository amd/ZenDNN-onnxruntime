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

#pragma once
#include "core/common/optional.h"
#include "core/providers/providers.h"
#include "core/providers/provider_factory_creators.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ArmNN(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CoreML(uint32_t);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptionsV2* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(const OrtDnnlProviderOptions* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Zendnn(const OrtZendnnProviderOptions* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(const OrtMIGraphXProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(
    uint32_t flags, const optional<std::string>& partitioning_stop_ops_list);
// std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tvm(const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_type, bool enable_vpu_fast_compile, const char* device_id, size_t num_of_threads, const char* cache_dir);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const OrtOpenVINOProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rknpu();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rocm(const OrtROCMProviderOptions* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptionsV2* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cann(const OrtCANNProviderOptions* provider_options);

// EP for internal testing
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_InternalTesting(
    const std::unordered_set<std::string>& supported_ops);

namespace test {

// unique_ptr providers with default values for session registration
std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider();
std::unique_ptr<IExecutionProvider> CudaExecutionProviderWithOptions(const OrtCUDAProviderOptionsV2* provider_options);
std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultZendnnExecutionProvider();
std::unique_ptr<IExecutionProvider> DnnlExecutionProviderWithOptions(const OrtDnnlProviderOptions* provider_options);
std::unique_ptr<IExecutionProvider> ZendnnExecutionProviderWithOptions(const OrtZendnnProviderOptions* provider_options);
// std::unique_ptr<IExecutionProvider> DefaultTvmExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider();
std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptions* params);
std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptionsV2* params);
std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider();
std::unique_ptr<IExecutionProvider> MIGraphXExecutionProviderWithOptions(const OrtMIGraphXProviderOptions* params);
std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultArmNNExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultRocmExecutionProvider(bool test_tunable_op = false);
std::unique_ptr<IExecutionProvider> DefaultCoreMLExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultSnpeExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultQnnExecutionProvider();
std::unique_ptr<IExecutionProvider> QnnExecutionProviderWithOptions(const ProviderOptions& options);
std::unique_ptr<IExecutionProvider> DefaultXnnpackExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultCannExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultDmlExecutionProvider();

std::unique_ptr<IExecutionProvider> DefaultInternalTestingExecutionProvider(
    const std::unordered_set<std::string>& supported_ops);

}  // namespace test
}  // namespace onnxruntime
