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

    std::unordered_set<std::string> binary_ops = {"Add", "Div", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Sub"};
    std::unordered_set<std::string> elementwise_ops = {"Abs", "Clip", "Elu", "Exp", "LeakyRelu", "Log", "Relu", "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"};
    std::unordered_set<std::string> pool_ops = {"MaxPool", "AveragePool", "GlobalAveragePool"};
    std::unordered_set<std::string> conv_fusion_ops = {"Conv", "ConvRelu", "ConvClip", "ConvElu", "ConvSwish", "ConvAddRelu", "ConvAdd"};
    zendnn_bf16_ops_map.insert("MatMul");
    zendnn_bf16_ops_map.insert("Reshape");
    zendnn_bf16_ops_map.insert("Concat");
    zendnn_bf16_ops_map.insert("Flatten");
    zendnn_bf16_ops_map.insert("LRN");
    zendnn_bf16_ops_map.insert("Transpose");
    zendnn_bf16_ops_map.insert(binary_ops.begin(), binary_ops.end());
    zendnn_bf16_ops_map.insert(elementwise_ops.begin(), elementwise_ops.end());
    zendnn_bf16_ops_map.insert(pool_ops.begin(), pool_ops.end());
    zendnn_bf16_ops_map.insert(conv_fusion_ops.begin(), conv_fusion_ops.end());
}

bool ZendnnBF16Ops::IsNodeBF16enabled() {
    bool zendnn_enable_bf16 = false;

    const std::string enable_bf16_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty())
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);
    return zendnn_enable_bf16;
}

bool ZendnnBF16Ops::IsNodeBF16enabled(std::string nodename) {

    bool zendnn_enable_bf16 = IsNodeBF16enabled();
    auto it = zendnn_bf16_ops_map.find(nodename);
    if( it != zendnn_bf16_ops_map.end() && zendnn_enable_bf16)
        return true;
    else
        return false;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
