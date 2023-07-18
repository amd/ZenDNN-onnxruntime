/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

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

#include "zendnn_execution_provider_info.h"

#include "core/providers/zendnn/zendnn_provider_options.h"
#include "core/framework/provider_options_utils.h"
#include "core/common/common.h"

namespace onnxruntime::zendnn::provider_option_names {
constexpr const char *kUseArena = "use_arena";
constexpr const char *kThreadpoolArgs = "threadpool_args";
}  // namespace onnxruntime::zendnn::provider_option_names

namespace onnxruntime {

ZendnnExecutionProviderInfo ZendnnExecutionProviderInfo::FromProviderOptions(
    const ProviderOptions &options) {
    ZendnnExecutionProviderInfo info{};
    ORT_THROW_IF_ERROR(
        ProviderOptionsParser{}
        .AddValueParser(
            zendnn::provider_option_names::kThreadpoolArgs,
    [&info](const std::string& value_str) -> Status {
        size_t address;
        ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
        info.threadpool_args = reinterpret_cast<void *>(address);
        return Status::OK();
    })
    .AddAssignmentToReference(zendnn::provider_option_names::kUseArena,
                              info.use_arena)
    .Parse(options));
    return info;
}

ProviderOptions ZendnnExecutionProviderInfo::ToProviderOptions(
    const ZendnnExecutionProviderInfo &info) {
    const ProviderOptions options{
        {zendnn::provider_option_names::kUseArena, MakeStringWithClassicLocale(info.use_arena)},
        {zendnn::provider_option_names::kThreadpoolArgs, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.threadpool_args))},
    };

    return options;
}

ProviderOptions ZendnnExecutionProviderInfo::ToProviderOptions(
    const OrtZendnnProviderOptions &info) {
    const ProviderOptions options{
        {zendnn::provider_option_names::kUseArena, MakeStringWithClassicLocale(info.use_arena)},
    };

    return options;
}

}  // namespace onnxruntime