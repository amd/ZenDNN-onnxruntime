/*******************************************************************************
* Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "zendnn.hpp"
#include "zendnn_bf16_ops.h"


namespace onnxruntime {
namespace ort_zendnn {

ZendnnBF16Ops::ZendnnBF16Ops() {
    zendnn_bf16_ops_map.insert("MatMul");
    zendnn_bf16_ops_map.insert("Binary");

    zendnn_bf16_ops_map.insert("Conv");
    zendnn_bf16_ops_map.insert("MaxPool");
    zendnn_bf16_ops_map.insert("AveragePool");

    zendnn_bf16_ops_map.insert("Relu");     //Support op
    zendnn_bf16_ops_map.insert("ConvRelu"); //Fusion op
}

bool ZendnnBF16Ops::IsZenDNNBF16enabled() {
    bool zendnn_enable_bf16 = false;

    const std::string enable_bf16_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty()) {
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);
    }
    return zendnn_enable_bf16;
}

bool ZendnnBF16Ops::isNodeBF16enabled(std::string nodename) {

    auto it = zendnn_bf16_ops_map.find(nodename);
    if (it != zendnn_bf16_ops_map.end()) {
        return true;
    }
    else {
        return false;
    }
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
