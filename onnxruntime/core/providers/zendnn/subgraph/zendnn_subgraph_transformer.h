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

#pragma once
#include "zendnn_subgraph.h"

namespace onnxruntime {
namespace ort_zendnn {

class ZendnnGraphTransformer {
  public:
    // The passed in onnx subgraph viewer is only valid during "Compile" phase,
    // so keep a reference to that onnx subgraph in ZendnnSubgraph is risky.
    // passed in the onnx subgraph viewer explicitly to make sure we manage the lifetime correctly.
    void Apply(ZendnnSubgraph &subgraph,
               const onnxruntime::GraphViewer &onnx_subgraph_viewer_);
    void OptimizeInplaceOps(ZendnnSubgraph &subgraph);
    ZendnnGraphTransformer() {
        const std::string debug_log_env =
            onnxruntime::GetEnvironmentVar("ORT_ZENDNN_DEBUG_LOG");
        if (!debug_log_env.empty()) {
            debug_log_ = (std::stoi(debug_log_env) == 0 ? false : true);
        }
    }

  private:
    void Gelu(ZendnnSubgraph &subgraph,
              const onnxruntime::GraphViewer &onnx_subgraph_viewer);
    void FastGelu(ZendnnSubgraph &subgraph,
                  const onnxruntime::GraphViewer &onnx_subgraph_viewer);
    bool FastGeluFirstFormula(ZendnnSubgraph &subgraph,
                              const onnxruntime::GraphViewer &onnx_subgraph_viewer, ZendnnNode *node,
                              int &fastgelu_index);
    void FastGeluSecondFormula(ZendnnSubgraph &subgraph,
                               const onnxruntime::GraphViewer &onnx_subgraph_viewer, ZendnnNode *node,
                               int &fastgelu_index);
    bool FastGeluFormulaCommon(ZendnnSubgraph &subgraph,
                               const onnxruntime::GraphViewer &onnx_subgraph_viewer,
                               ZendnnNode *gelu_start_node, int32_t x_input_index, ZendnnNode *tanh_node,
                               std::vector<size_t> &gelu_indices, int &fastgelu_index);
    bool IsInitilizedWithExpectedValue(const onnxruntime::GraphViewer
                                       &onnx_subgraph_viewer, ZendnnTensor &input_arg, float expected_value);
    void ConvRelu(ZendnnSubgraph &subgraph);
    void BatchnormRelu(ZendnnSubgraph &subgraph);
    void ConvClip(ZendnnSubgraph &subgraph);
    void QConvClip(ZendnnSubgraph &subgraph);
    void ConvElu(ZendnnSubgraph &subgraph);
    void ConvSwish(ZendnnSubgraph &subgraph);
    void MatMulBinaryEltwise(ZendnnSubgraph &subgraph);
    void RemoveMatMulIntegerZP(ZendnnSubgraph &subgraph,
                               const onnxruntime::GraphViewer &onnx_subgraph_viewer);
    void MatMulIntegerBinaryEltwise(ZendnnSubgraph &subgraph);
    // Function used to identify and fuse post ops
    //
    // @param[in] subgraph the ZendnnSubgrapy that we are searching for possible fusions
    // @param[in] node is the first node to check if it contains a binary or an elementwise op
    // @param[in/out] indicies list of all the indicies for the nodes that will be fused
    // @param[in/out] fused_node_inputs list of all the inputs that will be part of the fused node
    // @param[in/out] attr_node this node contains the attributes that will be passed onto the final fused node
    //
    // @return a pointer to the node after the last identified binary/elementwise fusion
    ZendnnNode *FuseBinaryEltwisePostOps(ZendnnSubgraph &subgraph, ZendnnNode *node,
                                         std::vector<size_t> &indices, std::vector<ZendnnTensor *> &fused_node_inputs,
                                         ZendnnNode *&attr_node);
    // This function checks a few things
    //   - the node in question has a single output
    //   - The output of the node is only consumed by a one other node
    //   - the output tensor from the node is going to another node within the subgraph
    // If all of the above is true this will return true. It will return false otherwise.
    //
    // It is possible for a node to fail one or more of the checks above and still be fusable.
    //
    // The name of the function was chosen because this check is required for most of the node fusions
    // found in the code.
    //
    // The last node in a fusion does not need to pass this check.
    bool IsNodeFusable(ZendnnSubgraph &subgraph, ZendnnNode *node) const;
    void ResolveFusion(ZendnnSubgraph &subgraph, std::vector<size_t> old_indices,
                       std::unique_ptr<ZendnnNode> new_node);
    bool ProduceGraphOutput(ZendnnSubgraph &subgraph, ZendnnNode &node);
    bool IsGraphOutput(ZendnnSubgraph &subgraph, ZendnnTensor &tensor);
    void ConvAddRelu(ZendnnSubgraph &subgraph);
    void processConvAddRelu(ZendnnSubgraph &subgraph, ZendnnNode *current_node,
                            ZendnnNode *next_node, size_t index);
    void optimizeForFusionCase(ZendnnSubgraph &subgraph);
    void optimizeForNonFusionCase(ZendnnSubgraph &subgraph);
    void FuseLN(ZendnnSubgraph &subgraph);
    /* Quantize Node specific graph optimizations. */
    void QConvRelu(ZendnnSubgraph &subgraph);
    void QConvAdd(ZendnnSubgraph &subgraph);
    bool debug_log_ = false;
};

}  // namespace ort_zendnn
}  // namespace onnxruntime
