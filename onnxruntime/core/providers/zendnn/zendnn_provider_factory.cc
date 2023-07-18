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

#include "core/providers/zendnn/zendnn_provider_factory.h"

#include <atomic>
#include <cassert>

#include "core/providers/shared_library/provider_api.h"

#include "core/providers/zendnn/zendnn_provider_factory_creator.h"
#include "core/providers/zendnn/zendnn_execution_provider.h"
#include "core/providers/zendnn/zendnn_execution_provider_info.h"

using namespace onnxruntime;

namespace onnxruntime {

struct ZendnnProviderFactory : IExecutionProviderFactory {
    ZendnnProviderFactory(const ZendnnExecutionProviderInfo &info) : info_(info) {}
    ~ZendnnProviderFactory() override {}

    std::unique_ptr<IExecutionProvider> CreateProvider() override;

  private:
    ZendnnExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> ZendnnProviderFactory::CreateProvider() {
    return std::make_unique<ZendnnExecutionProvider>(info_);
}

struct ProviderInfo_Zendnn_Impl : ProviderInfo_Zendnn {
    void ZendnnExecutionProviderInfo__FromProviderOptions(const ProviderOptions
            &options,
            ZendnnExecutionProviderInfo &info) override {
        info = ZendnnExecutionProviderInfo::FromProviderOptions(options);
    }

    std::shared_ptr<IExecutionProviderFactory>
    CreateExecutionProviderFactory(const ZendnnExecutionProviderInfo &info)
    override {
        return std::make_shared<ZendnnProviderFactory>(info);
    }
} g_info;

struct Zendnn_Provider : Provider {
    std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
        int use_arena) override {
#if defined(ZENDNN_OPENMP)
        LoadOpenMP();
#endif  // defined(ZENDNN_OPENMP) && defined(_WIN32)

        // Map options to provider info
        ZendnnExecutionProviderInfo info{};
        info.use_arena = use_arena;
        return std::make_shared<ZendnnProviderFactory>(info);
    }

    std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(
        const void *options) override {
#if defined(ZENDNN_OPENMP)
        LoadOpenMP();
#endif  // defined(ZENDNN_OPENMP) && defined(_WIN32)
        // Cast options
        auto zendnn_options = reinterpret_cast<const OrtZendnnProviderOptions *>
                              (options);

        // Map options to provider info
        ZendnnExecutionProviderInfo info{};
        info.use_arena = zendnn_options->use_arena;
        info.threadpool_args = zendnn_options->threadpool_args;

        return std::make_shared<ZendnnProviderFactory>(info);
    }

    void UpdateProviderOptions(void *provider_options,
                               const ProviderOptions &options) override {
        auto internal_options =
            onnxruntime::ZendnnExecutionProviderInfo::FromProviderOptions(options);
        auto &zendnn_options = *reinterpret_cast<OrtZendnnProviderOptions *>
                               (provider_options);

        zendnn_options.use_arena = internal_options.use_arena;
        zendnn_options.threadpool_args = internal_options.threadpool_args;
    }

    ProviderOptions GetProviderOptions(const void *provider_options) override {
        auto &options = *reinterpret_cast<const OrtZendnnProviderOptions *>
                        (provider_options);
        return ZendnnExecutionProviderInfo::ToProviderOptions(options);
    }

    void Initialize() override {
    }

    void Shutdown() override {
    }

    void *GetInfo() override {
        return &g_info;
    }

  private:
    void LoadOpenMP() {
#if defined(_WIN32)
        // We crash when unloading ZENDNN on Windows when OpenMP also unloads (As there are threads
        // still running code inside the openmp runtime DLL if OMP_WAIT_POLICY is set to ACTIVE).
        // To avoid this, we pin the OpenMP DLL so that it unloads as late as possible.
        HMODULE handle{};
#ifdef _DEBUG
        constexpr const char *dll_name = "vcomp140d.dll";
#else
        constexpr const char *dll_name = "vcomp140.dll";
#endif  // _DEBUG
        ::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, dll_name, &handle);
        assert(handle);  // It should exist
#endif               // defined(_WIN32)
    }

} g_provider;

}  // namespace onnxruntime

extern "C" {

    ORT_API(onnxruntime::Provider *, GetProvider) {
        return &onnxruntime::g_provider;
    }
}
