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

#include <memory>

#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "core/providers/zendnn/zendnn_provider_options.h"

namespace onnxruntime {
struct IExecutionProviderFactory;
struct ZendnnExecutionProviderInfo;

struct ProviderInfo_Zendnn {
    virtual void ZendnnExecutionProviderInfo__FromProviderOptions(
        const onnxruntime::ProviderOptions &options,
        onnxruntime::ZendnnExecutionProviderInfo &info) = 0;
    virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory>
    CreateExecutionProviderFactory(const onnxruntime::ZendnnExecutionProviderInfo
                                   &info) = 0;
};

}  // namespace onnxruntime
