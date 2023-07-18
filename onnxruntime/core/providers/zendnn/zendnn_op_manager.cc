/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright(C) 2021 Intel Corporation
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

#include "zendnn_op_manager.h"
#include <iostream>

namespace onnxruntime {
ZendnnOpManager::ZendnnOpManager() {
    zendnn_ops_map_.emplace(std::make_pair("Abs",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Add",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("AveragePool",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnPoolNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("BatchNormalization",
                                           std::unique_ptr<ZendnnNodeCapability>(new
                                                   ZendnnBatchNormalizationNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("BiasGelu",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Cast",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnCastNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Clip",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Concat",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnConcatNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Conv",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnDefaultNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("QLinearGlobalAveragePool",
                                           std::unique_ptr<ZendnnQLinearGlobalAveragePoolNodeCapability>
                                           (new ZendnnQLinearGlobalAveragePoolNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("QuantizeLinear",
                                           std::unique_ptr<ZendnnQuantizeLinearNodeCapability>(new
                                                   ZendnnQuantizeLinearNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("QLinearConv",
                                           std::unique_ptr<ZendnnQLinearConvNodeCapability>(new
                                                   ZendnnQLinearConvNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("QLinearAdd",
                                           std::unique_ptr<ZendnnQLinearBinaryNodeCapability>(new
                                                   ZendnnQLinearBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("DequantizeLinear",
                                           std::unique_ptr<ZendnnNodeCapability>(new
                                                   ZendnnDequantizeLinearNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Div",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("DynamicQuantizeLinear",
                                           std::unique_ptr<ZendnnNodeCapability>(new
                                                   ZendnnDynamicQuantizeLinearNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Elu",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Equal",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Erf",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnErfNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Exp",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("FastGelu",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Flatten",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnFlattenNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("FusedMatMul",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnMatMulNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Gelu",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Gemm",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnGemmNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("GlobalAveragePool",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnPoolNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("GlobalMaxPool",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnPoolNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Greater",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("GreaterOrEqual",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("LayerNormalization",
                                           std::unique_ptr<ZendnnNodeCapability>(new
                                                   ZendnnLayerNormalizationNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("LeakyRelu",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Less",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("LessOrEqual",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Log",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("LRN",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnLRNNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("MatMul",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnMatMulNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("MatMulInteger",
                                           std::unique_ptr<ZendnnNodeCapability>(new
                                                   ZendnnMatMulIntegerNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("MaxPool",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnPoolNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Mul",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Pow",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnPowNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("QAttention",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnQAttentionNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceL1",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceL2",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceLogSum",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceLogSumExp",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceMax",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceMean",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceMin",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceProd",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceSum",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReduceSumSquare",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReduceNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Relu",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Reshape",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnReshapeNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Round",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Sigmoid",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("SkipLayerNormalization",
                                           std::unique_ptr<ZendnnNodeCapability>(new
                                                   ZendnnSkipLayerNormalizationNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Softmax",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnSoftmaxNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Softplus",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Squeeze",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnSqueezeNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Sqrt",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Sub",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnBinaryNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Sum",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnSumNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Tanh",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnElementwiseCapability())));
    zendnn_ops_map_.emplace(std::make_pair("Transpose",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnDefaultNodeCapability({type_float32, type_bfloat16}))));
    zendnn_ops_map_.emplace(std::make_pair("Unsqueeze",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnSqueezeNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("VitisAIConv2D",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZenVitisAINodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("VitisAIConv2DWithSum",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZenVitisAINodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("VitisAIDepthwiseConv2D",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZenVitisAINodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("VitisAIConcatV2",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZenVitisAINodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("VitisAIMaxPool",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZenVitisAINodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("VitisAIAvgPool",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZenVitisAINodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("VitisAIConv2DWithoutBias",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZenVitisAINodeCapability())));
#if defined(ENABLE_TRAINING)
    zendnn_ops_map_.emplace(std::make_pair("AveragePoolGrad",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnPoolNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ConvGrad",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnDefaultNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("MaxPoolGrad",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnPoolNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("ReluGrad",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnDefaultNodeCapability())));
    zendnn_ops_map_.emplace(std::make_pair("SoftmaxGrad",
                                           std::unique_ptr<ZendnnNodeCapability>(new ZendnnSoftmaxNodeCapability())));
#endif  // ENABLE_TRAINING
}

bool ZendnnOpManager::IsNodeSupported(const Node *node,
                                      const GraphViewer &graph_viewer) const {
    auto it = zendnn_ops_map_.find(node->OpType());
    if (it == zendnn_ops_map_.end()) {
        return false;
    }
    return it->second->Supported(node, graph_viewer);
}

bool ZendnnOpManager::IsOpTypeAvalible(const std::string &opType) const {
    auto op_it = zendnn_ops_map_.find(opType);
    return (op_it != zendnn_ops_map_.end());
}
}  // namespace onnxruntime
