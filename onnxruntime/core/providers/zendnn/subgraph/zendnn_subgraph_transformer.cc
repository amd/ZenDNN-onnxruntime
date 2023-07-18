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

#include "zendnn_subgraph_transformer.h"
#include "core/providers/shared_library/provider_api.h"
#ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <memory>

namespace onnxruntime {
namespace ort_zendnn {

// apply all transformation rules in order
void ZendnnGraphTransformer::Apply(ZendnnSubgraph &subgraph,
                                   const onnxruntime::GraphViewer &onnx_subgraph_viewer) {
    bool enable_conv_relu_fusion = true;
    const std::string fusion_conv_relu_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_CONV_RELU_FUSION_ENABLE");
    if (!fusion_conv_relu_env.empty()) {
        enable_conv_relu_fusion = (std::stoi(fusion_conv_relu_env) == 0 ? false : true);
    }
    if (enable_conv_relu_fusion) {
        ConvRelu(subgraph);
    }

    bool enable_conv_clip_fusion = false;
    const std::string fusion_conv_clip_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_CONV_CLIP_FUSION_ENABLE");
    if (!fusion_conv_clip_env.empty()) {
        enable_conv_clip_fusion = (std::stoi(fusion_conv_clip_env) == 0 ? false : true);
    }
    if (enable_conv_clip_fusion) {
        ConvClip(subgraph);
    }

    bool enable_bn_relu_fusion = false;
    const std::string fusion_bn_relu_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_BN_RELU_FUSION_ENABLE");
    if (!fusion_bn_relu_env.empty()) {
        enable_bn_relu_fusion = (std::stoi(fusion_bn_relu_env) == 0 ? false : true);
    }
    if (enable_bn_relu_fusion) {
        BatchnormRelu(subgraph);
    }

    bool enable_matmul_binary_eltwise = true;
    const std::string enable_matmul_binary_eltwise_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_ENABLE_MATMUL_BINARY_ELTWISE");
    if (!enable_matmul_binary_eltwise_env.empty()) {
        enable_matmul_binary_eltwise = (std::stoi(enable_matmul_binary_eltwise_env) == 0
                                        ? false : true);
    }
    if (enable_matmul_binary_eltwise) {
        MatMulBinaryEltwise(subgraph);
    }

    bool enable_gelu = true;
    const std::string enable_gelu_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_ENABLE_GELU");
    if (!enable_gelu_env.empty()) {
        enable_gelu = (std::stoi(enable_gelu_env) == 0 ? false : true);
    }
    if (enable_gelu) {
        Gelu(subgraph, onnx_subgraph_viewer);
    }

    bool enable_fast_gelu = true;
    const std::string enable_fast_gelu_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_ENABLE_FAST_GELU");
    if (!enable_fast_gelu_env.empty()) {
        enable_fast_gelu = (std::stoi(enable_fast_gelu_env) == 0 ? false : true);
    }
    if (enable_fast_gelu) {
        FastGelu(subgraph, onnx_subgraph_viewer);
    }

    bool remove_matmul_integer = true;
    const std::string remove_matmul_integer_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_REMOVE_MATMUL_INTEGER");
    if (!remove_matmul_integer_env.empty()) {
        remove_matmul_integer = (std::stoi(remove_matmul_integer_env) == 0 ? false :
                                 true);
    }
    if (remove_matmul_integer) {
        RemoveMatMulIntegerZP(subgraph, onnx_subgraph_viewer);
    }

    bool matmul_integer_binary_eltwise = true;
    const std::string matmul_integer_binary_eltwise_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_MATMUL_INTEGER_BINARY_ELTWISE");
    if (!matmul_integer_binary_eltwise_env.empty()) {
        matmul_integer_binary_eltwise = (std::stoi(matmul_integer_binary_eltwise_env) ==
                                         0 ? false : true);
    }
    if (matmul_integer_binary_eltwise) {
        MatMulIntegerBinaryEltwise(subgraph);
    }

    bool enable_conv_elu_fusion_ = false;
    // by default conv-elu fusion is false, use flag ZENDNN_CONV_ELU_FUSION_ENABLE to toggle this fusion
    const std::string fusion_conv_elu_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_CONV_ELU_FUSION_ENABLE");
    if (!fusion_conv_elu_env.empty()) {
        enable_conv_elu_fusion_ = (std::stoi(fusion_conv_elu_env) == 0 ? false : true);
    }
    if (enable_conv_elu_fusion_) {
        ConvElu(subgraph);
    }

    bool enable_conv_add_fusion_ = false;
    const std::string fusion_conv_add_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_CONV_ADD_FUSION_ENABLE");
    if (!fusion_conv_add_env.empty()) {
        enable_conv_add_fusion_ = (std::stoi(fusion_conv_add_env) == 0 ? false : true);
    }
    if (enable_conv_add_fusion_) {
        ConvAddRelu(subgraph);
    }

    /* Resnet strides trick optimization is dependent on existing graph.
    * CONV_ADD_FUSION will modify the represented graph in ZenDNN-EP.
    * Changes : Conv->Add->Relu(3nodes) = Conv-Add-Relu(single node)
    * Conv->Add(2nodes) = Conv-Add(single node)
    * So, different implementations exist for Fusion and Non-Fusion cases.
    */
    const std::string strides_optimization =
        onnxruntime::GetEnvironmentVar("ZENDNN_RESNET_STRIDES_OPT1_ENABLE");
    if (!strides_optimization.empty() &&
            (std::stoi(strides_optimization) == 0 ? false : true)) {
        if (enable_conv_add_fusion_) {
            optimizeForFusionCase(subgraph);
        }
        else {
            optimizeForNonFusionCase(subgraph);
        }
    }

    bool ln_block_fusion = false;
    // by default ln_block fusion is false. Use flag ZENDNN_LN_FUSION_ENABLE to toggle this function
    const std::string fusion_ln =
        onnxruntime::GetEnvironmentVar("ZENDNN_LN_FUSION_ENABLE");
    if (!fusion_ln.empty()) {
        ln_block_fusion = (std::stoi(fusion_ln) == 0 ? false : true);
    }
    if (ln_block_fusion) {
        FuseLN(subgraph);
    }

    bool enable_qconv_relu_fusion = true;
    const std::string fusion_qconv_relu_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_QUANTIZE_CONV_RELU_FUSION_ENABLE");
    if (!fusion_qconv_relu_env.empty()) {
        enable_qconv_relu_fusion = (std::stoi(fusion_qconv_relu_env) == 0 ? false :
                                    true);
    }
    if (enable_qconv_relu_fusion) {
        QConvRelu(subgraph);
    }

    bool enable_qconv_add_fusion_ = false;
    const std::string fusion_qconv_add_env =
        onnxruntime::GetEnvironmentVar("ZENDNN_QUANTIZE_CONV_ADD_FUSION_ENABLE");
    if (!fusion_qconv_add_env.empty()) {
        enable_qconv_add_fusion_ = (std::stoi(fusion_qconv_add_env) == 0 ? false :
                                    true);
    }
    if (enable_qconv_add_fusion_) {
        QConvAdd(subgraph);
    }

    bool fusion_conv_swish = false;
    const std::string fusion_conv_swish_enable =
        onnxruntime::GetEnvironmentVar("ZENDNN_CONV_SWISH_FUSION_ENABLE");
    if (!fusion_conv_swish_enable.empty()) {
        fusion_conv_swish = (std::stoi(fusion_conv_swish_enable) == 0 ? false : true);
    }
    if (fusion_conv_swish) {
        ConvSwish(subgraph);
    }
}

//apply all inplace optimizations rules in order
void ZendnnGraphTransformer::OptimizeInplaceOps(ZendnnSubgraph &subgraph) {
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);

        if (zendnn_node == nullptr) {
            continue;
        }

        if (zendnn_node->OpType() == "Concat") {
            bool isInplace=true;
            for (auto &inp : zendnn_node->Inputs()) {
                if (inp->GetProducer().Exists() == false) {
                    isInplace = false;
                    break;
                }
                /* *********************************************************************
                ** Input tensors to Concat should be only 1(which is Concat op itself).
                ** If there are more, we cannot perform inplace concatenation operation.
                ** Here we remove the current concat Op and associated i/p tensors.
                ** *********************************************************************/
                if (inp->GetProducer().GetNode()->OpType() != "Conv" &&
                        inp->GetProducer().GetNode()->OpType() != "ConvRelu" &&
                        inp->GetProducer().GetNode()->OpType() != "ConvAdd" &&
                        inp->GetProducer().GetNode()->OpType() != "ConvAddRelu") {
                    isInplace = false;
                    break;
                }
                if (inp->GetConsumersCount() > 1) {
                    isInplace = false;
                    break;
                }
            }

            if (!isInplace) {
                continue;
            }
            /* At this point, all Input nodes to Concat are expected Conv nodes. */
            for (auto &inp : zendnn_node->Inputs()) {
                inp->GetProducer().GetNode()->isInplaceMemoryNode = true;
                inp->GetProducer().GetNode()->Attributes().insert(zendnn_node->Attributes());
            }
            // replace/rename existing node(concat) to ZendnnInception (zenInceptionIPC for now)
            zendnn_node->setOpType("zenInceptionIPC");
        }
    }
}

//Check and Do optimization of Conv+Add for QConvAdd. QLinearConv + QLinearAdd
void ZendnnGraphTransformer::QConvAdd(ZendnnSubgraph &subgraph) {
    /* Prepare QLinearConv and QLinearAdd nodes for QConvAdd.
     * Scales and zero-points have to be propagated and modified accordingly.
     */

    static int qconv_add_index = 0;
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto qconv_node = subgraph.GetZendnnNode(index);
        if (qconv_node == nullptr) {
            continue;
        }
        if (qconv_node->OpType() != "QLinearConv") {
            continue;
        }

        if (qconv_node->OutputCount() != 1) {
            ORT_THROW("Invalid Conv node");
        }

        if (qconv_node->Output(0).Exists() &&
                qconv_node->Output(0).GetConsumersCount() != 1) {
            continue;
        }

        auto qadd_node = qconv_node->Output(0).GetConsumers()[0].GetNode();
        if (qadd_node == nullptr) {
            continue;
        }

        if (qadd_node->OpType() != "QLinearAdd") {
            continue;
        }

        int qadd_input_count = 0;
        for (auto input_tensor : qadd_node->Inputs()) {
            if (input_tensor->GetProducer().Exists()) {
                auto input_node = input_tensor->GetProducer().GetNode();
                if (input_node == nullptr) {
                    continue;
                }
                auto input_node_type = input_node->OpType();
                if (input_node_type == "QLinearConv" ||
                        input_node_type == "QConvAdd" ||
                        input_node_type == "QConvAdd_v1" ||
                        input_node_type == "QConvAddRelu" ||
                        input_node_type == "MaxPool") {
                    qadd_input_count++;
                }
            }
        }

        if (qadd_input_count != 2) {
            continue;
        }

        auto qadd_inputs = qadd_node->Inputs();
        //add is taking two inputs from the same conv output
        //not sure if ZenDNN would support such post ops
        if (qadd_inputs[0] == qadd_inputs[3] ||
                qadd_inputs[0]->Dim() != qadd_inputs[3]->Dim()) {
            continue;
        }

        /* Inputs to qadd from previous QConv nodes are at index 0 and 3 respectively. */
        std::vector<size_t> idxes{0, 3};
        for (auto idx : idxes) {
            auto &qConvInp = qadd_node->Input((int)idx);
            auto qnode = qConvInp.GetProducer().GetNode();

            if (qnode->OpType() == "MaxPool") {
                qnode = qnode->Input(0).GetProducer().GetNode();
            }

            if (qnode->OpType() == "QLinearConv") {
                qnode->OpType() = "QLinearConv_v1";
                /* Check for QLinearConv as one of the children of the current QLinearConv with other child as QLinearAdd*/
                /* do qnode->OpType() = "QLinearConv_v2". mobilenet ConvAdd optimization. */
                for (auto cons : qnode->Output(0).GetConsumers()) {
                    auto intConvNode = cons.GetNode();
                    if (intConvNode->OpType() == "QLinearConv") {
                        intConvNode->OpType() = "QLinearConv_v2";
                    }
                }
            }
            else if (qnode->OpType() == "QConvAdd" || qnode->OpType() == "QConvAddRelu") {
                qnode->OpType() = "QConvAdd_v1";
                for (auto cons : qnode->Output(0).GetConsumers()) {
                    auto intConvNode = cons.GetNode();
                    if (intConvNode->OpType() == "QLinearConv") {
                        intConvNode->OpType() = "QLinearConv_v2";
                    }
                }
            }
        }

        auto fused_node_inputs = qconv_node->Inputs();

        //the 3rd input to fused conv
        if (qconv_node->Output(0).Name() == qadd_inputs[0]->Name()) {
            fused_node_inputs.push_back(qadd_inputs[3]);  //Tensor memory
            fused_node_inputs.push_back(qadd_inputs[4]);  //sum_scale
        }
        else {
            fused_node_inputs.push_back(qadd_inputs[0]);  //Tensor memory
            fused_node_inputs.push_back(qadd_inputs[1]);  //sum_scale
        }

        fused_node_inputs.push_back(qadd_inputs[6]);  //add_out_scale
        fused_node_inputs.push_back(qadd_inputs[7]);  //add_out_zp

        auto fused_node_output = qadd_node->Outputs()[0];
        auto fused_node_name = qconv_node->Name() + "_QConvAdd_" + std::to_string(
                                   qconv_add_index);
        auto fused_node_type = "QConvAdd";
        std::vector<size_t> fusableNodes = {qconv_node->Index(), qadd_node->Index()};

        ZendnnNode *ar_node;
        if (qadd_node->Output(0).Exists() &&
                qadd_node->Output(0).GetConsumersCount() == 1) {
            /* ar => Add followed by Relu*/
            ar_node = qadd_node->Output(0).GetConsumers()[0].GetNode();
            if ((ar_node != nullptr) && (ar_node->OpType() == "Relu")) {
                fused_node_type = "QConvAddRelu";
                fused_node_output = ar_node->Outputs()[0];
                fusableNodes.push_back(ar_node->Index());
            }
        }

        //construct new node
        auto fused_node = std::make_unique<ZendnnNode>();
        fused_node->Name() = fused_node_name;
        fused_node->OpType() = fused_node_type;
        fused_node->Inputs() = fused_node_inputs;

        fused_node->Outputs() = {fused_node_output};
        fused_node->Attributes().insert(qconv_node->Attributes());

        LOGS_DEFAULT(INFO) <<
                           "#########################  ##################### fuse ["<< qconv_node->Name()
                           << "] and [" << qadd_node->Name() << "] into " <<fused_node_type;
        ResolveFusion(subgraph, fusableNodes, std::move(fused_node));
    }
}

//resolve a fusion by replacing old_indices nodes with a new_node
//unneeded tensors will be deleted, old news' edges will be cleared
//new_node will be set with new edges and inserted to subgraph
void ZendnnGraphTransformer::ResolveFusion(ZendnnSubgraph &subgraph,
        std::vector<size_t> old_indices, std::unique_ptr<ZendnnNode> new_node) {
    //the tensors to keep
    std::unordered_set<std::string> keep_tensors;

    //get keep tensors from new_node
    //all tensors related to new_node needs to be kept
    for (auto input : new_node->Inputs()) {
        if (input && input->Exists()) {
            keep_tensors.insert(input->Name());
        }
    }

    for (auto output : new_node->Outputs()) {
        if (output && output->Exists()) {
            keep_tensors.insert(output->Name());
        }
    }

    //find out tensors to remove, cleanup tensor consumers and producer
    std::unordered_set<std::string> tensors_to_remove;
    for (auto index : old_indices) {
        auto cur_node = subgraph.GetZendnnNode(index);
        {
            int input_index = 0;
            for (auto input : cur_node->Inputs()) {
                if (input && input->Exists()) {
                    input->RemoveConsumer(ZendnnNodeArg(cur_node, input_index, false));
                    if (!keep_tensors.count(input->Name())) {
                        tensors_to_remove.insert(input->Name());
                    }
                }
                input_index++;
            }
        }
        for (auto output : cur_node->Outputs()) {
            if (output && output->Exists()) {
                output->ResetProducer();
                if (!keep_tensors.count(output->Name())) {
                    tensors_to_remove.insert(output->Name());
                }
            }
        }
    }

    //remove unused tensors
    for (const auto &tensor_name : tensors_to_remove) {
        auto tensor = subgraph.GetZendnnTensor(tensor_name);
        if (tensor) {
            //has consumer and producer
            if (tensor->GetConsumers().size() || tensor->GetProducer().Exists()) {
                continue;
            }
            else {
                subgraph.RemoveTensor(tensor_name);
            }
        }
        //subgraph.RemoveTensor(tensor_name);
    }
    //remove unused nodes
    for (auto index : old_indices) {
        subgraph.RemoveNode(index);
    }

    //reestablish producer and consumer for tensors related to new node
    //such tensors should not get deleted
    {
        size_t input_index = 0;
        for (auto input : new_node->Inputs()) {
            if (input) {
                input->AddConsumer(ZendnnNodeArg(new_node.get(), input_index, false));
            }

            input_index++;
        }

        size_t output_index = 0;
        for (auto output : new_node->Outputs()) {
            if (output) {
                output->SetProducer(ZendnnNodeArg(new_node.get(), output_index, true));
            }
            output_index++;
        }
    }

    //new node now has correct input output tensors as well as tensor connections
    //subgraph now owns the new node
    subgraph.AddNode(std::move(new_node));
}

//helper to determine whether a tensor acts as subgraph output
bool ZendnnGraphTransformer::IsGraphOutput(ZendnnSubgraph &subgraph,
        ZendnnTensor &tensor) {
    auto graph_outputs = subgraph.GetZendnnOutputs();
    if (std::find(graph_outputs.cbegin(), graph_outputs.cend(),
                  &tensor) != graph_outputs.cend()) {
        return true;
    }
    return false;
}

//helper to determien whether
bool ZendnnGraphTransformer::ProduceGraphOutput(ZendnnSubgraph &subgraph,
        ZendnnNode &node) {
    auto graph_outputs = subgraph.GetZendnnOutputs();
    for (auto output : node.Outputs()) {
        if (output && output->Exists()) {
            if (IsGraphOutput(subgraph, *output)) {
                return true;
            }
        }
    }
    return false;
}


bool ZendnnGraphTransformer::IsNodeFusable(ZendnnSubgraph &subgraph,
        ZendnnNode *node) const {
    if (node == nullptr) {
        return false;
    }
    //isSingleOutput(ZendnnNode* node);
    if (node->OutputCount() != 1) {
        std::string s = "Invalid " + node->OpType() + " node";
        ORT_THROW(s);
    }
    //isConsumedBySingleNode(ZendnnNode* node);
    if (node->Output(0).Exists() && node->Output(0).GetConsumers().size() != 1) {
        return false;
    }
    //isOutputPartOfSubgraph(ZendnnSubgraph& subgraph, ZendnnNode* node);
    auto graph_outputs = subgraph.GetZendnnOutputs();
    if (std::find(graph_outputs.cbegin(), graph_outputs.cend(),
                  &node->Output(0)) != graph_outputs.cend()) {
        return false;
    }
    return true;
}

bool IsScalar(const ZendnnTensor &input_arg) {
    auto dim = input_arg.Dim();
    auto dim_size = dim.size();
    return dim_size == 0 || (dim_size == 1 && dim[0] == 1);
}

bool ZendnnGraphTransformer::IsInitilizedWithExpectedValue(
    const onnxruntime::GraphViewer &onnx_subgraph_viewer, ZendnnTensor &input_arg,
    float expected_value) {
    if (!IsScalar(input_arg)) {
        return false;
    }

    const ONNX_NAMESPACE::TensorProto *tensor_proto = nullptr;
    if (!onnx_subgraph_viewer.GetInitializedTensor(input_arg.Name(),
            tensor_proto)) {
        return false;
    }

    if (tensor_proto == nullptr) {
        return false;
    }

    if (!tensor_proto->has_raw_data()) {
        return false;
    }

    const auto data_type = input_arg.Type();
    if (data_type == zendnn::memory::data_type::f32) {
        const float *val = reinterpret_cast<const float *>
                           (tensor_proto->raw_data().data());
        if (std::isnan(val[0]) || std::isinf(val[0])) {
            if (std::isinf(val[0]) && std::isinf(expected_value) &&
                    (std::signbit(val[0]) == std::signbit(expected_value))) {
                return true;
            }
            return false;
        }

        const float atol = 1e-8f;
        const float rtol = 1e-5f;
        float diff = std::abs(val[0] - expected_value);
        if (diff > (atol + rtol * std::abs(expected_value))) {
            return false;
        }
    }
    else {
        // Not expected data types.
        return false;
    }

    return true;

}

ZendnnNode *FirstParentByType(ZendnnNode *node,
                              const std::string &parent_type) {
    for (size_t i = 0; i < node->InputCount(); ++i) {
        auto prev_node = node->Input(static_cast<int>(i)).GetProducer().GetNode();
        if (prev_node != nullptr && prev_node->OpType() == parent_type) {
            return prev_node;
        }
    }
    return nullptr;
}

/*
     This function fuses subgraph like the following into one Gelu node.
     Subgraph pattern 1:
                   +-------Mul(0.5)---------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul ==>
                          (B=1.4142...)        (1)

      Subgraph pattern 2:
                   +------------------------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
                          (B=1.4142...)        (1)            (0.5)

       After Fusion:
                [root]--> Gelu ==>
*/
void ZendnnGraphTransformer::Gelu(ZendnnSubgraph &subgraph,
                                  const onnxruntime::GraphViewer &onnx_subgraph_viewer) {
    static int gelu_index = 0;
    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto div_node = subgraph.GetZendnnNode(index);
        std::vector<size_t> gelu_indices;
        //----------------------------
        if (div_node == nullptr || div_node->OpType() != "Div") {
            continue;
        }

        // Check second input is sqrt(2)
        // Some Bert models uses this approximation of SQRT2 in the Gelu function
        float approximated_sqrt_two = 1.4142099618911743f;
        if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, div_node->Input(1),
                                           approximated_sqrt_two) &&
                !IsInitilizedWithExpectedValue(onnx_subgraph_viewer, div_node->Input(1),
                                               static_cast<float>(M_SQRT2))) {
            continue;
        }

        if (!IsNodeFusable(subgraph, div_node)) {
            continue;
        }
        gelu_indices.push_back(div_node->Index());
        //----------------------------
        auto erf_node = div_node->Output(0).GetConsumers()[0].GetNode();
        if (erf_node == nullptr || erf_node->OpType() != "Erf") {
            continue;
        }

        if (!IsNodeFusable(subgraph, erf_node)) {
            continue;
        }
        gelu_indices.push_back(erf_node->Index());
        //----------------------------
        auto add_node = erf_node->Output(0).GetConsumers()[0].GetNode();
        if (add_node == nullptr || add_node->OpType() != "Add") {
            continue;
        }

        bool is_add_input0 = add_node->Input(0).Name() == erf_node->Output(0).Name();
        if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                           add_node->Input(is_add_input0 ? 1 : 0), 1.0f)) {
            continue;
        }

        if (!IsNodeFusable(subgraph, add_node)) {
            continue;
        }
        gelu_indices.push_back(add_node->Index());
        //----------------------------
        auto mul1_node = add_node->Output(0).GetConsumers()[0].GetNode();
        if (mul1_node == nullptr || mul1_node->OpType() != "Mul") {
            continue;
        }

        //if (!IsNodeFusable(subgraph, mul1_node)) {
        //  continue;
        //}
        gelu_indices.push_back(mul1_node->Index());
        //----------------------------
        // look for Mul(0.5) using pattern 1 shown above
        bool is_pattern_1 = true;
        auto mul2_node = FirstParentByType(mul1_node, "Mul");
        if (mul2_node != nullptr) {
            // the input Div and Mul2 should share at least one input.
            bool is_mul2_input0 = div_node->Input(0).Name() == mul2_node->Input(0).Name();
            bool is_mul2_input1 = div_node->Input(0).Name() == mul2_node->Input(1).Name();
            if (!(is_mul2_input0 ^ is_mul2_input1)) {
                is_pattern_1 = false;
            }
            if (is_pattern_1 &&
                    !IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                                   mul2_node->Input(is_mul2_input0 ? 1 : 0), 0.5f)) {
                is_pattern_1 = false;
            }
            if (is_pattern_1 && !IsNodeFusable(subgraph, mul2_node)) {
                is_pattern_1 = false;
            }
        }
        else {
            is_pattern_1 = false;
        }

        // look for Mul(0.5) using pattern 2 shown above
        if (!is_pattern_1) {
            // We only need to check mul1_node IsNodeFusable for pattern 2
            if (!IsNodeFusable(subgraph, mul1_node)) {
                continue;
            }
            mul2_node = mul1_node->Output(0).GetConsumers()[0].GetNode();
            if (mul2_node == nullptr || mul2_node->OpType() != "Mul") {
                continue;
            }

            if (mul2_node->OutputCount() != 1) {
                ORT_THROW("Invalid Mul node");
            }
            bool is_mul2_first_input = mul2_node->Input(0).Name() == mul1_node->Output(
                                           0).Name();
            if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                               mul2_node->Input(is_mul2_first_input ? 1 : 0), 0.5f)) {
                continue;
            }
        }
        gelu_indices.push_back(mul2_node->Index());

        //construct new node
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = div_node->Name() + "_Gelu_" + std::to_string(gelu_index++);
        new_node->OpType() = "Gelu";
        new_node->Inputs().push_back(div_node->Inputs()[0]);
        if (is_pattern_1) {
            for (auto def : mul1_node->Outputs()) {
                new_node->Outputs().push_back(def);
            }
        }
        else {
            for (auto def : mul2_node->Outputs()) {
                new_node->Outputs().push_back(def);
            }
        }
        // no attributes needed for Gelu if needed this can be updated
        //new_node->Attributes().insert(div_node->Attributes());

        //insert new node, remove original nodes, connect new edges
        ResolveFusion(subgraph, {gelu_indices}, std::move(new_node));
        if (debug_log_) {
            LOGS_DEFAULT(ERROR) << "Gelu fusion found [" << gelu_index << "]";
        }
    }
}

/*
Rewrite graph fusing Gelu activation subgraph to a single Gelu node.
The formula corresponding to Gelu activation subgraph :

    x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))) or
    x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))),

where x is the input.
*/
void ZendnnGraphTransformer::FastGelu(ZendnnSubgraph &subgraph,
                                      const onnxruntime::GraphViewer &onnx_subgraph_viewer) {
    static int fastgelu_index = 0;
    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);
        if (!FastGeluFirstFormula(subgraph, onnx_subgraph_viewer, zendnn_node,
                                  fastgelu_index)) {
            FastGeluSecondFormula(subgraph, onnx_subgraph_viewer, zendnn_node,
                                  fastgelu_index);
        }
    }
}

/*
Rewrite graph fusing Gelu activation subgraph to a single Gelu node.
The formula corresponding to Gelu activation subgraph :

    x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))

where x is the input.
*/
bool ZendnnGraphTransformer::FastGeluFirstFormula(ZendnnSubgraph &subgraph,
        const onnxruntime::GraphViewer &onnx_subgraph_viewer, ZendnnNode *mul1_node,
        int &fastgelu_index) {
    std::vector<size_t> gelu_indices;
    //----------mul(0.44715)------------------
    if (mul1_node == nullptr || mul1_node->OpType() != "Mul") {
        return false;
    }
    int32_t mul1_input_index = -1;
    const float mul_val = 0.044715f;
    for (auto i = 0; i < 2; i++) {
        if (IsInitilizedWithExpectedValue(onnx_subgraph_viewer, mul1_node->Input(i),
                                          mul_val)) {
            mul1_input_index = i;
            break;
        }
    }
    if (mul1_input_index == -1) {
        return false;
    }

    if (!IsNodeFusable(subgraph, mul1_node)) {
        return false;
    }
    gelu_indices.push_back(mul1_node->Index());
    //-------------Mul---------------
    auto mul2_node = mul1_node->Output(0).GetConsumers()[0].GetNode();
    if (mul2_node == nullptr || mul2_node->OpType() != "Mul") {
        return false;
    }
    if (!IsNodeFusable(subgraph, mul2_node)) {
        return false;
    }
    gelu_indices.push_back(mul2_node->Index());
    //-------------Add(1.0)---------------
    auto add1_node = mul2_node->Output(0).GetConsumers()[0].GetNode();
    if (add1_node == nullptr || add1_node->OpType() != "Add") {
        return false;
    }
    bool is_add_input0 = mul2_node->Output(0).Name() == add1_node->Input(0).Name();
    if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                       add1_node->Input(is_add_input0 ? 1 : 0), 1.0f)) {
        return false;
    }

    if (!IsNodeFusable(subgraph, add1_node)) {
        return false;
    }
    gelu_indices.push_back(add1_node->Index());
    //-------------Mul---------------
    auto mul3_node = add1_node->Output(0).GetConsumers()[0].GetNode();
    if (mul3_node == nullptr || mul3_node->OpType() != "Mul") {
        return false;
    }
    if (!IsNodeFusable(subgraph, mul3_node)) {
        return false;
    }
    gelu_indices.push_back(mul3_node->Index());
    //-------------Mul(0.7978845834732056f)---------------
    auto prev_mul4_node = FirstParentByType(mul3_node, "Mul");
    if (prev_mul4_node == nullptr) {
        return false;
    }

    int32_t mul4_input_index = -1;
    const float mul4_val = 0.7978845834732056f;
    for (auto i = 0; i < 2; i++) {
        if (IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                          prev_mul4_node->Input(i), mul4_val)) {
            mul4_input_index = i;
            break;
        }
    }
    if (mul4_input_index == -1) {
        return false;
    }

    if (!IsNodeFusable(subgraph, prev_mul4_node)) {
        return false;
    }
    gelu_indices.push_back(prev_mul4_node->Index());

    auto tanh_node = mul3_node->Output(0).GetConsumers()[0].GetNode();
    int32_t x_input_index = (mul1_input_index == 0) ? 1 : 0;
    if (FastGeluFormulaCommon(subgraph, onnx_subgraph_viewer, mul1_node,
                              x_input_index, tanh_node, gelu_indices, fastgelu_index)) {
        if (debug_log_) {
            LOGS_DEFAULT(ERROR) << "FastGelu fusion found [" << fastgelu_index <<
                                "] (first formula)";
        }
        return true;
    }
    return false;
}

/*
Rewrite graph fusing Gelu activation subgraph to a single Gelu node.
The formula corresponding to Gelu activation subgraph :

    x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))),

where x is the input.
*/
void ZendnnGraphTransformer::FastGeluSecondFormula(ZendnnSubgraph &subgraph,
        const onnxruntime::GraphViewer &onnx_subgraph_viewer, ZendnnNode *pow_node,
        int &fastgelu_index) {
    std::vector<size_t> gelu_indices;
    //---------Pow-------------------
    if (pow_node == nullptr || pow_node->OpType() != "Pow") {
        return;
    }

    auto &pow_exponent = pow_node->Input(1);
    if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer, pow_exponent, 3.0f)) {
        return;
    }

    if (!IsNodeFusable(subgraph, pow_node)) {
        return;
    }
    gelu_indices.push_back(pow_node->Index());
    //----------Mul(0.044714998453855515f)------------------
    auto mul1_node = pow_node->Output(0).GetConsumers()[0].GetNode();
    if (mul1_node == nullptr || mul1_node->OpType() != "Mul") {
        return;
    }

    float fastgelu_muliplyer = 0.044714998453855515f;
    bool is_mul1_input0 = pow_node->Output(0).Name() == mul1_node->Input(0).Name();
    if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                       mul1_node->Input(is_mul1_input0 ? 1 : 0), fastgelu_muliplyer)) {
        return;
    }
    if (!IsNodeFusable(subgraph, mul1_node)) {
        return;
    }
    gelu_indices.push_back(mul1_node->Index());
    //----------Add------------------
    auto add1_node = mul1_node->Output(0).GetConsumers()[0].GetNode();
    if (add1_node == nullptr || add1_node->OpType() != "Add") {
        return;
    }

    if (!IsNodeFusable(subgraph, add1_node)) {
        return;
    }
    gelu_indices.push_back(add1_node->Index());
    //----------Mul(sqrt(2/pi))------------------
    auto mul2_node = add1_node->Output(0).GetConsumers()[0].GetNode();
    if (mul2_node == nullptr || mul2_node->OpType() != "Mul") {
        return;
    }

    // constant is sqrt(2/pi)
    float fastgelu_sqrt_2_div_pi = 0.7978845834732056f;
    bool is_mul2_input0 = add1_node->Output(0).Name() == mul2_node->Input(0).Name();
    if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                       mul2_node->Input(is_mul2_input0 ? 1 : 0), fastgelu_sqrt_2_div_pi)) {
        return;
    }

    if (!IsNodeFusable(subgraph, mul2_node)) {
        return;
    }
    gelu_indices.push_back(mul2_node->Index());

    //----------Tanh------------------
    auto tanh_node = mul2_node->Output(0).GetConsumers()[0].GetNode();
    // since the first node is pow the x_input_index is always 0
    if (FastGeluFormulaCommon(subgraph, onnx_subgraph_viewer, pow_node, 0,
                              tanh_node, gelu_indices, fastgelu_index)) {
        if (debug_log_) {
            LOGS_DEFAULT(ERROR) << "FastGelu fusion found [" << fastgelu_index <<
                                "] (second formula)";
        }
    }
}

/*
 Looks for the part of FastGelu that is common to both formulas if the pattern is found
 return true otherwise return false.
    i.e. x * 0.5 * (1.0 + tanh(...))

    x * 0.5 * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))) or
    x * 0.5 * (1.0 + tanh((sqrt(2 / pi) * (x + 0.044715 * pow(x, 3))))),
where x is the input.
*/
bool ZendnnGraphTransformer::FastGeluFormulaCommon(ZendnnSubgraph &subgraph,
        const onnxruntime::GraphViewer &onnx_subgraph_viewer,
        ZendnnNode *gelu_start_node, int32_t x_input_index, ZendnnNode *tanh_node,
        std::vector<size_t> &gelu_indices, int &fastgelu_index) {
    //----------Tanh------------------
    if (tanh_node == nullptr || tanh_node->OpType() != "Tanh") {
        return false;
    }

    if (!IsNodeFusable(subgraph, tanh_node)) {
        return false;
    }
    gelu_indices.push_back(tanh_node->Index());
    //----------Add(1.0)------------------
    auto add2_node = tanh_node->Output(0).GetConsumers()[0].GetNode();
    if (add2_node == nullptr || add2_node->OpType() != "Add") {
        return false;
    }
    bool is_add2_input0 = tanh_node->Output(0).Name() == add2_node->Input(0).Name();
    if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                       add2_node->Input(is_add2_input0 ? 1 : 0), 1.0f)) {
        return false;
    }

    if (!IsNodeFusable(subgraph, add2_node)) {
        return false;
    }
    gelu_indices.push_back(add2_node->Index());
    //----------Mul------------------
    auto mul3_node = add2_node->Output(0).GetConsumers()[0].GetNode();
    if (mul3_node == nullptr || mul3_node->OpType() != "Mul") {
        return false;
    }

    if (mul3_node->OutputCount() != 1) {
        ORT_THROW("Invalid Mul node");
    }

    gelu_indices.push_back(mul3_node->Index());
    //---------Mul(0.5)---------------------
    if (mul3_node->InputCount() != 2) {
        return false;
    }
    auto prev_mul4_node = FirstParentByType(mul3_node, "Mul");
    if (prev_mul4_node == nullptr) {
        return false;
    }
    bool is_mul_input0 = gelu_start_node->Input(x_input_index).Name() ==
                         prev_mul4_node->Input(0).Name();
    bool is_mul_input1 = gelu_start_node->Input(x_input_index).Name() ==
                         prev_mul4_node->Input(1).Name();
    if (!(is_mul_input0 ^ is_mul_input1)) {
        return false;
    }
    if (!IsInitilizedWithExpectedValue(onnx_subgraph_viewer,
                                       prev_mul4_node->Input(is_mul_input0 ? 1 : 0), 0.5f)) {
        return false;
    }
    if (!IsNodeFusable(subgraph, prev_mul4_node)) {
        return false;
    }
    gelu_indices.push_back(prev_mul4_node->Index());

    //construct new node
    auto new_node = std::make_unique<ZendnnNode>();
    new_node->Name() = "Zendnn_FastGelu_" + std::to_string(fastgelu_index++);
    new_node->OpType() = "FastGelu";
    new_node->Inputs().push_back(gelu_start_node->Inputs()[x_input_index]);
    for (auto def : mul3_node->Outputs()) {
        new_node->Outputs().push_back(def);
    }
    // No Attributes needed for FastGelu. If they are needed this can be added in.
    //new_node->Attributes().insert(gelu_start_node->Attributes());

    //insert new node, remove original nodes, connect new edges
    ResolveFusion(subgraph, {gelu_indices}, std::move(new_node));
    return true;
}

void ZendnnGraphTransformer::ConvRelu(ZendnnSubgraph &subgraph) {
    //global index of convrelu
    static int conv_relu_index = 0;

    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);

        //look for conv relu pattern
        if (zendnn_node == nullptr) {
            continue;
        }

        if (zendnn_node->OpType() != "Conv") {
            continue;
        }

        if (!IsNodeFusable(subgraph, zendnn_node)) {
            continue;
        }

        auto next_zendnn_node = zendnn_node->Output(0).GetConsumers()[0].GetNode();
        if (next_zendnn_node == nullptr) {
            continue;
        }
        if (next_zendnn_node->OpType() != "Relu" &&
                next_zendnn_node->OpType() != "LeakyRelu") {
            continue;
        }

        //construct new node
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = zendnn_node->Name() + "_ConvRelu_" + std::to_string(
                               conv_relu_index++);
        new_node->OpType() = "ConvRelu";
        for (auto def : zendnn_node->Inputs()) {
            new_node->Inputs().push_back(def);
        }
        for (auto def : next_zendnn_node->Outputs()) {
            new_node->Outputs().push_back(def);
        }
        new_node->Attributes().insert(zendnn_node->Attributes());
        new_node->Attributes().insert(next_zendnn_node->Attributes());

        //insert new node, remove original nodes, connect new edges
        if (debug_log_) {
            LOGS_DEFAULT(ERROR) << "ConvRelu fusion of [" << zendnn_node->Name() <<
                                "] and [" << next_zendnn_node->Name() << "]";
        }
        ResolveFusion(subgraph, {zendnn_node->Index(), next_zendnn_node->Index()},
                      std::move(new_node));
    }
}

void ZendnnGraphTransformer::QConvRelu(ZendnnSubgraph &subgraph) {
    //global index of convrelu
    static int qconv_relu_index = 0;

    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);

        //look for conv relu pattern
        if (zendnn_node == nullptr) {
            continue;
        }

        if (zendnn_node->OpType() != "QLinearConv") {
            continue;
        }

        if (!IsNodeFusable(subgraph, zendnn_node)) {
            continue;
        }

        auto next_zendnn_node = zendnn_node->Output(0).GetConsumers()[0].GetNode();
        if (next_zendnn_node == nullptr) {
            continue;
        }
        if (next_zendnn_node->OpType() != "Relu") {
            continue;
        }

        //construct new node
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = zendnn_node->Name() + "_QConvRelu_" + std::to_string(
                               qconv_relu_index++);
        new_node->OpType() = "QConvRelu";
        for (auto def : zendnn_node->Inputs()) {
            new_node->Inputs().push_back(def);
        }
        for (auto def : next_zendnn_node->Outputs()) {
            new_node->Outputs().push_back(def);
        }
        new_node->Attributes().insert(zendnn_node->Attributes());

        //insert new node, remove original nodes, connect new edges
        if (debug_log_) {
            LOGS_DEFAULT(ERROR) << "QConvRelu fusion of [" << zendnn_node->Name() <<
                                "] and [" << next_zendnn_node->Name() << "]";
        }
        ResolveFusion(subgraph, {zendnn_node->Index(), next_zendnn_node->Index()},
                      std::move(new_node));
    }
}

void ZendnnGraphTransformer::MatMulBinaryEltwise(ZendnnSubgraph &subgraph) {
    static int fused_index = 0;
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        std::vector<size_t> matmul_binary_eltwize_indices = {};
        auto zendnn_node = subgraph.GetZendnnNode(index);
        auto attr_node = zendnn_node;

        if (zendnn_node == nullptr || zendnn_node->OpType() != "MatMul") {
            continue;
        }

        if (!IsNodeFusable(subgraph, zendnn_node)) {
            continue;
        }
        auto fused_node_inputs = zendnn_node->Inputs();
        matmul_binary_eltwize_indices.push_back(zendnn_node->Index());

        zendnn_node = FuseBinaryEltwisePostOps(subgraph,
                                               zendnn_node,
                                               matmul_binary_eltwize_indices,
                                               fused_node_inputs,
                                               attr_node);

        if (!(matmul_binary_eltwize_indices.size() > 1)) {
            matmul_binary_eltwize_indices.clear();
            continue;
        }

        //construct new node
        auto fused_node = std::make_unique<ZendnnNode>();
        fused_node->Name() = "MatMulPostOps_fusion" + std::to_string(fused_index++);
        std::string fused_node_name = "MatMulPostOps";
        for (size_t i : matmul_binary_eltwize_indices) {
            if (subgraph.GetZendnnNode(i)->OpType() != "MatMul") {
                fused_node->AppendPostOp(subgraph.GetZendnnNode(i)->OpType());
            }
        }
        fused_node->OpType() = fused_node_name;
        fused_node->Inputs() = fused_node_inputs;
        fused_node->Outputs() = {zendnn_node->Outputs()[0]};

        fused_node->Attributes().insert(attr_node->Attributes());

        if (debug_log_) {
            std::stringstream ss;
            for (size_t i : matmul_binary_eltwize_indices) {
                ss << subgraph.GetZendnnNode(i)->OpType() << "[" << subgraph.GetZendnnNode(
                       i)->Name() << "] ";
            }
            LOGS_DEFAULT(ERROR) << fused_node->OpType() << "[" << fused_node->Name() <<
                                "] fusion of " << ss.str();
        }
        // insert new node, remove original nodes, connect new edges
        ResolveFusion(subgraph, matmul_binary_eltwize_indices, std::move(fused_node));
    }
}

void ZendnnGraphTransformer::RemoveMatMulIntegerZP(ZendnnSubgraph &subgraph,
        const onnxruntime::GraphViewer &onnx_subgraph_viewer) {
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);

        //look for matmulint
        if (zendnn_node == nullptr || zendnn_node->OpType() != "MatMulInteger") {
            continue;
        }

        //if B zero point exists
        if (!(zendnn_node->InputCount() >= 4 && zendnn_node->Input(3).Exists())) {
            continue;
        }

        auto &b_zero_point = zendnn_node->Input(3);
        const ONNX_NAMESPACE::TensorProto *tensor_proto = nullptr;
        if (!onnx_subgraph_viewer.GetInitializedTensor(b_zero_point.Name(),
                tensor_proto)) {
            continue;
        }

        if (tensor_proto == nullptr) {
            continue;
        }

        const auto &dims = tensor_proto->dims();
        auto dim_size = tensor_proto->dims_size();
        int num_elements = 1;
        for (int i = 0; i < dim_size; i++) {
            num_elements *= int(dims[i]);
        }

        //check if b_zp is all zeros, assume data is s8 since only s8 weight is supported in zendnn
        bool all_zero = true;
        std::vector<int8_t> unpacked_tensor;
        unpacked_tensor.resize(num_elements,1);
        ORT_THROW_IF_ERROR(onnxruntime::utils::UnpackTensor(*tensor_proto,
                           tensor_proto->has_raw_data() ? tensor_proto->raw_data().data() : nullptr,
                           tensor_proto->has_raw_data() ? tensor_proto->raw_data().size() : 0,
                           reinterpret_cast<int8_t *>(unpacked_tensor.data()), num_elements));
        for (const auto &val : unpacked_tensor) {
            if (val != 0) {
                all_zero = false;
                break;
            }
        }


        if (!all_zero) {
            continue;
        }

        if (debug_log_) {
            LOGS_DEFAULT(ERROR) << "Remove weight ZP of [" << zendnn_node->Name() << "]";
        }
        //remove b_zero_point's consumer matmulint
        b_zero_point.RemoveConsumer(ZendnnNodeArg(zendnn_node, 3, false));
        //detach b_zero_point from matmulint node
        zendnn_node->Inputs()[3] = nullptr;
    }
}

void ZendnnGraphTransformer::MatMulIntegerBinaryEltwise(
    ZendnnSubgraph &subgraph) {
    static int fused_index = 0;
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        std::vector<size_t> matmul_binary_eltwize_indices = {};
        auto zendnn_node = subgraph.GetZendnnNode(index);
        auto attr_node = zendnn_node;

        if (zendnn_node == nullptr || zendnn_node->OpType() != "MatMulInteger") {
            continue;
        }

        if (!IsNodeFusable(subgraph, zendnn_node)) {
            continue;
        }

        auto fused_node_inputs = zendnn_node->Inputs();
        matmul_binary_eltwize_indices.push_back(zendnn_node->Index());

        auto next_zendnn_node = zendnn_node->Output(0).GetConsumers()[0].GetNode();
        if (next_zendnn_node == nullptr) {
            continue;
        }
        if (next_zendnn_node->OpType() != "Cast") {
            continue;
        }

        if (!IsNodeFusable(subgraph, next_zendnn_node)) {
            continue;
        }

        zendnn_node = next_zendnn_node;
        matmul_binary_eltwize_indices.push_back(zendnn_node->Index());

        zendnn_node = FuseBinaryEltwisePostOps(subgraph,
                                               zendnn_node,
                                               matmul_binary_eltwize_indices,
                                               fused_node_inputs,
                                               attr_node);

        if (!(matmul_binary_eltwize_indices.size() > 1)) {
            matmul_binary_eltwize_indices.clear();
            continue;
        }

        //construct new node
        auto fused_node = std::make_unique<ZendnnNode>();
        fused_node->Name() = "MatMulIntegerPostOps_fusion" + std::to_string(
                                 fused_index++);
        std::string fused_node_name = "MatMulIntegerPostOps";
        for (size_t i : matmul_binary_eltwize_indices) {
            if (subgraph.GetZendnnNode(i)->OpType() != "MatMulInteger" &&
                    subgraph.GetZendnnNode(i)->OpType() != "Cast") {
                fused_node->AppendPostOp(subgraph.GetZendnnNode(i)->OpType());
            }
        }
        fused_node->OpType() = fused_node_name;
        fused_node->Inputs() = fused_node_inputs;
        fused_node->Outputs() = {zendnn_node->Outputs()[0]};

        fused_node->Attributes().insert(attr_node->Attributes());

        if (debug_log_) {
            std::stringstream ss;
            for (size_t i : matmul_binary_eltwize_indices) {
                ss << subgraph.GetZendnnNode(i)->OpType() << "[" << subgraph.GetZendnnNode(
                       i)->Name() << "] ";
            }
            LOGS_DEFAULT(ERROR) << fused_node->OpType() << "[" << fused_node->Name() <<
                                "] fusion of " << ss.str();
        }
        // insert new node, remove original nodes, connect new edges
        ResolveFusion(subgraph, matmul_binary_eltwize_indices, std::move(fused_node));
    }
}

ZendnnNode *ZendnnGraphTransformer::FuseBinaryEltwisePostOps(
    ZendnnSubgraph &subgraph,
    ZendnnNode *node,
    std::vector<size_t> &indices,
    std::vector<ZendnnTensor *> &fused_node_inputs,
    ZendnnNode *&attr_node) {
    std::unordered_set<std::string> binary_ops = {"Add", "Div", "Mul", "Sub"};
    std::unordered_set<std::string> elementwise_ops = {"Abs", "Elu", "Exp", "LeakyRelu", "Log", "Relu",
                                                       "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"
                                                      };
    // Upto 32 post-ops are supported by ZenDNN framework.
    const size_t MAX_POST_OP_COUNT = 32;
    bool attribute_flag = false;
    auto zendnn_node = node;
    size_t post_op_count = 0;
    while (post_op_count < MAX_POST_OP_COUNT) {
        if (!IsNodeFusable(subgraph, zendnn_node)) {
            break;
        }

        auto next_zendnn_node = zendnn_node->Output(0).GetConsumers()[0].GetNode();
        if (next_zendnn_node == nullptr) {
            break;
        }
        auto next_type = next_zendnn_node->OpType();
        bool is_binary_op = !(binary_ops.count(next_type) == 0);
        bool is_eltwise_op = !(elementwise_ops.count(next_type) == 0);
        if (!is_binary_op && !is_eltwise_op) {
            break;
        }

        if (is_binary_op) {
            if (zendnn_node->Output(0).Name() == next_zendnn_node->Inputs()[0]->Name()) {
                fused_node_inputs.push_back(next_zendnn_node->Inputs()[1]);
            }
            else {
                // ZenDNN can only fuse binary post op for the second input. Due to the fact
                // that division and subraction are not associative we can not fuse them if the
                // input for that node is the first input.
                if (next_zendnn_node->OpType() == "Div" ||
                        next_zendnn_node->OpType() == "Sub") {
                    break;
                }
                fused_node_inputs.push_back(next_zendnn_node->Inputs()[0]);
            }
        }
        else if (is_eltwise_op) {
            // We only support a single node with an "alpha" attribute. If we see Elu or
            // LeakyRelu set attr_node and attribute_flag. If the attribute_flag is set break
            // out of the while loop looking for additional post ops. Since the next op cannot
            // be part of the postop fusion.
            if (next_zendnn_node->OpType() == "Elu" ||
                    next_zendnn_node->OpType() == "LeakyRelu") {
                if (attribute_flag) {
                    break;
                }
                attr_node = next_zendnn_node;
                attribute_flag = true;
            }
        }
        zendnn_node = next_zendnn_node;
        indices.push_back(zendnn_node->Index());
        post_op_count++;
    }
    return zendnn_node;
}

void ZendnnGraphTransformer::ConvSwish(ZendnnSubgraph &subgraph) {
    //global index of convswish
    static int conv_swish_index = 0;
    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);
        //look for conv elu pattern
        if (zendnn_node == nullptr) {
            continue;
        }
        if (zendnn_node->OpType() != "Conv") {
            continue;
        }
        if (zendnn_node->OutputCount() != 1) {
            ORT_THROW("Invalid Conv node");
        }
        if (zendnn_node->Output(0).Exists() &&
                zendnn_node->Output(0).GetConsumers().size() != 2) {
            continue;
        }
        //check whether conv node's only output tensor is outputting the subgraph
        auto graph_outputs = subgraph.GetZendnnOutputs();
        if (std::find(graph_outputs.cbegin(), graph_outputs.cend(),
                      &zendnn_node->Output(0)) != graph_outputs.cend()) {
            continue;
        }

        auto next_zendnn_node = zendnn_node->Output(0).GetConsumers()[0].GetNode();
        if (next_zendnn_node == nullptr) {
            continue;
        }
        if ((next_zendnn_node->OpType() != "Mul") &&
                (next_zendnn_node->OpType() != "Sigmoid")) {
            continue;
        }


        auto mul_node_in_pattern = next_zendnn_node;
        auto sigmoid_node_in_pattern = next_zendnn_node;

        // if the next node is Mul, then the other output node must be sigmoid
        if (next_zendnn_node->OpType() == "Mul") {
            auto next_zendnn_node_1 = zendnn_node->Output(0).GetConsumers()[1].GetNode();
            if (next_zendnn_node_1->OpType() != "Sigmoid") {
                continue;
            }
            sigmoid_node_in_pattern = next_zendnn_node_1;
        }
        // if the next node is Sigmoid, then the other output node must be Mul
        if (next_zendnn_node->OpType() == "Sigmoid") {
            auto next_zendnn_node_1 = zendnn_node->Output(0).GetConsumers()[1].GetNode();
            if (next_zendnn_node_1->OpType() != "Mul") {
                continue;
            }
            mul_node_in_pattern = next_zendnn_node_1;
        }

        // For Mul, the second input must be sigmoid
        if (sigmoid_node_in_pattern->Output(0).GetConsumersCount() != 1 ||
                mul_node_in_pattern != sigmoid_node_in_pattern->Output(
                    0).GetConsumers()[0].GetNode()) {
            continue;
        }

        //construct new node
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = zendnn_node->Name() + "_ConvSwish_" + std::to_string(
                               conv_swish_index++);
        new_node->OpType() = "ConvSwish";
        for (auto def : zendnn_node->Inputs()) {
            new_node->Inputs().push_back(def);
        }
        for (auto def : mul_node_in_pattern->Outputs()) {
            new_node->Outputs().push_back(def);
        }
        new_node->Attributes().insert(zendnn_node->Attributes());

        new_node->Attributes().insert(zendnn_node->Output(
                                          0).GetConsumers()[0].GetNode()->Attributes());
        new_node->Attributes().insert(zendnn_node->Output(
                                          0).GetConsumers()[1].GetNode()->Attributes());
        //insert new node, remove original nodes, connect new edges
        ResolveFusion(subgraph, {zendnn_node->Index(), zendnn_node->Output(0).GetConsumers()[0].GetNode()->Index(), zendnn_node->Output(0).GetConsumers()[1].GetNode()->Index()},
                      std::move(new_node));
        LOGS_DEFAULT(INFO) << "fuse [" << zendnn_node << "] and [" << next_zendnn_node
                           << "] into ConvSwish";
    }
}

void ZendnnGraphTransformer::ConvElu(ZendnnSubgraph &subgraph) {
    //global index of convelu
    static int conv_elu_index = 0;
    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);
        //look for conv elu pattern
        if (zendnn_node == nullptr) {
            continue;
        }
        if (zendnn_node->OpType() != "Conv") {
            continue;
        }
        if (zendnn_node->OutputCount() != 1) {
            ORT_THROW("Invalid Conv node");
        }
        if (zendnn_node->Output(0).Exists() &&
                zendnn_node->Output(0).GetConsumers().size() != 1) {
            continue;
        }
        //check whether conv node's only output tensor is outputting the subgraph
        {
            auto graph_outputs = subgraph.GetZendnnOutputs();
            if (std::find(graph_outputs.cbegin(), graph_outputs.cend(),
                          &zendnn_node->Output(0)) != graph_outputs.cend()) {
                continue;
            }
        }
        auto next_zendnn_node = zendnn_node->Output(0).GetConsumers()[0].GetNode();
        if (next_zendnn_node == nullptr) {
            continue;
        }
        if (next_zendnn_node->OpType() != "Elu") {
            continue;
        }

        //construct new node
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = zendnn_node->Name() + "_ConvElu_" + std::to_string(
                               conv_elu_index++);
        new_node->OpType() = "ConvElu";
        for (auto def : zendnn_node->Inputs()) {
            new_node->Inputs().push_back(def);
        }
        for (auto def : next_zendnn_node->Outputs()) {
            new_node->Outputs().push_back(def);
        }
        new_node->Attributes().insert(zendnn_node->Attributes());
        //In order to get alpha attribute from ConvElu node, we need to
        //copy all attributes of present "Elu" node into the fused "ConvElu" node.
        new_node->Attributes().insert(next_zendnn_node->Attributes());
        //insert new node, remove original nodes, connect new edges
        ResolveFusion(subgraph, {zendnn_node->Index(), next_zendnn_node->Index()},
                      std::move(new_node));
        LOGS_DEFAULT(INFO) << "fuse [" << zendnn_node << "] and [" << next_zendnn_node
                           << "] into ConvElu";
    }
}

size_t get_input_count(ZendnnNode *add_node) {
    int input_count = 0;
    for (auto input_tensor : add_node->Inputs()) {
        if (input_tensor->GetProducer().Exists()) {
            auto input_tensor_type = input_tensor->GetProducer().GetNode()->OpType();
            if (input_tensor_type == "Conv" ||
                    input_tensor_type == "ConvAddRelu" ||
                    input_tensor_type == "ConvAdd" ||
                    input_tensor_type == "MaxPool") {
                input_count++;
            }
        }
    }
    return input_count;
}

/*
Checking case like -

     +-------->Conv-------+
     |                    |
     |                    V
 Node+------------------>Add

If this Conv-Add fusion happens then the fused node
will take 2 inputs from Node. Thus this fusion shouldn't happen.
*/
bool check_same_input(ZendnnNode *add_node, ZendnnNode *conv_node) {
    for (size_t i = 0; i < add_node->InputCount(); i++) {
        auto add_input_node = add_node->Input((int)i).GetProducer().GetNode();
        if (add_input_node != conv_node) {
            for (size_t j = 0; j < add_input_node->Output(0).GetConsumers().size(); j++) {
                auto consumer = add_input_node->Output(0).GetConsumers()[j].GetNode();
                if (consumer == conv_node) {
                    return false;
                }
            }
        }
    }
    return true;
}

void ZendnnGraphTransformer::ConvAddRelu(ZendnnSubgraph &subgraph) {
    static int conv_add_index = 0;
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto conv_node = subgraph.GetZendnnNode(index);
        if (conv_node == nullptr) {
            continue;
        }
        if (conv_node->OpType() != "Conv") {
            continue;
        }
        if (conv_node->Input(0).GetProducer().Exists() &&
                conv_node->Input(0).GetProducer().GetNode()->OpType() == "ConvAddRelu") {
            continue;
        }
        if (conv_node->OutputCount() != 1) {
            ORT_THROW("Invalid Conv node");
        }

        if (conv_node->Output(0).Exists() &&
                conv_node->Output(0).GetConsumers().size() != 1) {
            continue;
        }

        auto add_node = conv_node->Output(0).GetConsumers()[0].GetNode();
        if (add_node == nullptr) {
            continue;
        }
        if (add_node->OpType() != "Add") {
            continue;
        }

        if (get_input_count(add_node) != 2) {
            continue;
        }

        if (!check_same_input(add_node, conv_node)) {
            continue;
        }

        processConvAddRelu(subgraph, conv_node, add_node, conv_add_index++);
    }
}

bool isNextNodeRelu(ZendnnSubgraph &subgraph, ZendnnNode &conv_add_node) {
    if (conv_add_node.OutputCount() != 1) {
        return false;
    }

    if (conv_add_node.Output(0).Exists() &&
            conv_add_node.Output(0).GetConsumers().size() != 1) {
        return false;
    }

    {
        auto graph_outputs = subgraph.GetZendnnOutputs();
        if (std::find(graph_outputs.cbegin(), graph_outputs.cend(),
                      &conv_add_node.Output(0)) != graph_outputs.cend()) {
            return false;
        }
    }

    auto relu_node = conv_add_node.Output(0).GetConsumers()[0].GetNode();
    if (relu_node == nullptr) {
        return false;
    }

    if (relu_node->OpType() != "Relu") {
        return false;
    }

    return true;
}

void ZendnnGraphTransformer::processConvAddRelu(ZendnnSubgraph &subgraph,
        ZendnnNode *conv_node, ZendnnNode *add_node, size_t index) {
    auto conv_output_name = conv_node->Output(0).Name();
    auto add_inputs = add_node->Inputs();
    //add is taking two inputs from the same conv output
    //not sure if ZenDNN would support such post ops
    if (add_inputs[0] == add_inputs[1] ||
            add_inputs[0]->Dim() != add_inputs[1]->Dim()) {
        return;
    }
    auto fused_node_inputs = conv_node->Inputs();
    //the 3rd input to fused conv
    if (conv_output_name == add_inputs[0]->Name()) {
        fused_node_inputs.push_back(add_inputs[1]);
    }
    else {
        fused_node_inputs.push_back(add_inputs[0]);
    }
    auto fused_node_output = add_node->Outputs()[0];
    auto fused_node_name = conv_node->Name() + "_" + conv_node->OpType() +
                           add_node->OpType() + "_" + std::to_string(index);
    auto fused_node_type = conv_node->OpType() + add_node->OpType();

    //construct new node
    auto fused_node = std::make_unique<ZendnnNode>();
    fused_node->Name() = fused_node_name;
    fused_node->OpType() = fused_node_type;
    fused_node->Inputs() = fused_node_inputs;
    fused_node->Outputs() = {fused_node_output};
    fused_node->Attributes().insert(conv_node->Attributes());

    bool isNextRelu = isNextNodeRelu(subgraph, *fused_node);

    if (!isNextRelu) {
        LOGS_DEFAULT(INFO) << "fuse [" << conv_node->Name() << "] and [" <<
                           add_node->Name() << "] into ConvAdd";
        ResolveFusion(subgraph, {conv_node->Index(), add_node->Index()}, std::move(
                          fused_node));
    }
    else {
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = fused_node->Name() + "_" + "ConvAddRelu" + "_" +
                           std::to_string(index);
        new_node->OpType() = "ConvAddRelu";
        for (auto def : fused_node->Inputs()) {
            new_node->Inputs().push_back(def);
        }
        auto relu_node = fused_node->Output(0).GetConsumers()[0].GetNode();
        for (auto def : relu_node->Outputs()) {
            new_node->Outputs().push_back(def);
        }
        new_node->Attributes().insert(fused_node->Attributes());
        LOGS_DEFAULT(INFO) << "fuse [" << conv_node->Name() << "] , [" <<
                           add_node->Name() << "] and [" << relu_node->Name() << "] into ConvAddRelu";
        ResolveFusion(subgraph, {conv_node->Index(), add_node->Index(), relu_node->Index()},
                      std::move(new_node));
    }
}

Status GetStridesAttr(const ONNX_NAMESPACE::AttributeProto &proto,
                      std::vector<int64_t> &values) {
    ORT_RETURN_IF_NOT(
        proto.type() ==
        ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS,
        "proto.type() != AttributeProto_AttributeType_INTS"
    );
    values.reserve(proto.ints_size());
    for (int i = 0; i < proto.ints_size(); i++) {
        values.push_back(proto.ints(i));
    }
    return Status::OK();
}

void ZendnnGraphTransformer::optimizeForFusionCase(ZendnnSubgraph &subgraph) {
    if (subgraph.GetZendnnNodes().size() < 8) {
        return;
    }

    size_t index = 0;
    bool patternFound = true;
    while (patternFound) {
        auto tempZenNodes = subgraph.GetZendnnNodes();
        for (auto zendnn_node : tempZenNodes) {
            patternFound = false;
            index += 1;
            if (zendnn_node == nullptr) {
                continue;
            }

            // 4 inputs because the Conv node has 3 inputs X,W,B and Add had 1 extra input other than conv
            if (!(zendnn_node->OpType() == "ConvAddRelu" &&
                    zendnn_node->InputCount() == 4)) {
                continue;
            }

            auto n1_index = zendnn_node->Input(0).GetProducer().GetNode()->Index();
            auto n2_index = zendnn_node->Input(3).GetProducer().GetNode()->Index();

            if (!((subgraph.GetZendnnNode(n1_index)->OpType() == "ConvRelu") &&
                    (subgraph.GetZendnnNode(n2_index)->OpType() == "Conv")) ||
                    ((subgraph.GetZendnnNode(n2_index)->OpType() == "ConvRelu") &&
                     (subgraph.GetZendnnNode(n1_index)->OpType() == "Conv"))) {
                continue;
            }

            auto attr = subgraph.GetZendnnNode(n1_index)->Attributes().find("strides");
            std::vector<int64_t> n1strides;
            if (attr != subgraph.GetZendnnNode(n1_index)->Attributes().end()) {
                auto &proto = attr->second();
                Status status = GetStridesAttr(proto, n1strides);
            }
            else {
                continue;
            }

            attr = subgraph.GetZendnnNode(n2_index)->Attributes().find("strides");
            std::vector<int64_t> n2strides;
            if (attr != subgraph.GetZendnnNode(n2_index)->Attributes().end()) {
                auto &proto = attr->second();
                Status status = GetStridesAttr(proto, n2strides);
            }
            else {
                continue;
            }

            /* Either n1 or n2 should be 2,2 and 1,1 mutually. Else exit. */
            if (!((n1strides == std::vector<int64_t> {1, 1}) &&
                    (n2strides == std::vector<int64_t> {2, 2})) ||
                    ((n1strides == std::vector<int64_t> {2, 2}) &&
                     (n2strides == std::vector<int64_t> {1, 1}))) {
                continue;
            }

            /* Expecting node1 strides={2, 2} and node2 strides={1, 1}. Else swap for convention.*/
            if (n1strides == std::vector<int64_t> {1, 1}) {
                auto temp = n1_index;
                n1_index = n2_index;
                n2_index = temp;
            }

            auto reluParentID = subgraph.GetZendnnNode(n1_index)->Input(
                                    0).GetProducer().GetNode()->Index();
            auto _reluParentID = n2_index;
            auto n2p = _reluParentID;

            /* According to the pattern, this node should be available at 3 nodes above. */
            for (int i=0; i<2; i++) {
                n2p = _reluParentID;
                _reluParentID = subgraph.GetZendnnNode(_reluParentID)->Input(
                                    0).GetProducer().GetNode()->Index();
            }

            if (reluParentID != _reluParentID) {
                continue;
            }

            attr = subgraph.GetZendnnNode(n2p)->Attributes().find("strides");
            std::vector<int64_t> n2pstrides;
            if (attr != subgraph.GetZendnnNode(n2p)->Attributes().end()) {
                auto &proto = attr->second();
                Status status = GetStridesAttr(proto, n2pstrides);
            }
            else {
                continue;
            }

            if (n2pstrides != std::vector<int64_t> {2, 2}) {
                continue;
            }

            if (!((subgraph.GetZendnnNode(reluParentID)->OpType() == "ConvAddRelu") &&
                    (subgraph.GetZendnnNode(reluParentID)->InputCount() == 4))) {
                continue;;
            }

            attr = subgraph.GetZendnnNode(reluParentID)->Attributes().find("strides");
            std::vector<int64_t> n3strides;
            if (attr != subgraph.GetZendnnNode(reluParentID)->Attributes().end()) {
                auto &proto = attr->second();
                Status status = GetStridesAttr(proto, n3strides);
            }
            else {
                continue;
            }

            if (n3strides != std::vector<int64_t> {1, 1}) {
                continue;
            }

            auto an_p1ID = subgraph.GetZendnnNode(reluParentID)->Input(
                               0).GetProducer().GetNode()->Index();
            auto an_p2ID = subgraph.GetZendnnNode(reluParentID)->Input(
                               3).GetProducer().GetNode()->Index();

            /* Either an_p1ID or an_p2ID should be "Conv" and "Relu" mutually. Else exit. */
            if (!(((subgraph.GetZendnnNode(an_p1ID)->OpType() == "ConvAddRelu") &&
                    (subgraph.GetZendnnNode(an_p2ID)->OpType() == "ConvRelu")) ||
                    ((subgraph.GetZendnnNode(an_p1ID)->OpType() == "ConvRelu") &&
                     (subgraph.GetZendnnNode(an_p2ID)->OpType() == "ConvAddRelu")))) {
                continue;
            }

            /* Expecting an_p1ID = "Conv-Add-Relu" and an_p2ID="Conv-Relu". Else swap for convention.*/
            if (subgraph.GetZendnnNode(an_p1ID)->OpType() == "ConvRelu") {
                auto temp = an_p1ID;
                an_p1ID = an_p2ID;
                an_p2ID = temp;
            }

            auto rootReluParentID = an_p1ID;
            auto n4_index = subgraph.GetZendnnNode(an_p2ID)->Input(
                                0).GetProducer().GetNode()->Index();

            if (subgraph.GetZendnnNode(n4_index)->Input(0).GetProducer().GetNode()->Index()
                    != rootReluParentID) {
                continue;
            }

            /* If we have reached till here, we have found a pattern to insert MaxPool node.
            * Force n1 to strides to {1, 1}, n2p strides to {1, 1}, reluParentID/an_p2ID strides to {2, 2}.
            * Insert MaxPool node after rootReluParentID and before reluParentID
            */

            subgraph.GetZendnnNode(n2p)->SetStridesOpt(1);
            subgraph.GetZendnnNode(n1_index)->SetStridesOpt(1);
            subgraph.GetZendnnNode(an_p2ID)->SetStridesOpt(2);

            auto dummyMaxPool = std::make_unique<ZendnnNode>();
            dummyMaxPool->Name() = "MaxPool_" + std::to_string(index);
            dummyMaxPool->OpType() = "MaxPool";
            dummyMaxPool->SetStridesOpt(2);

            auto dummyMaxPool_output_tensor_name = "DMaxPool_" + std::to_string(
                    index) + "_Y";
            auto dummyMaxPool_output_tensor = std::make_unique<ZendnnTensor>
                                              (dummyMaxPool_output_tensor_name);

            subgraph.InsertNode(subgraph.GetZendnnNode(rootReluParentID),
                                subgraph.GetZendnnNode(reluParentID), std::move(dummyMaxPool),
                                std::move(dummyMaxPool_output_tensor));

            patternFound = true;
            break;
        }
    }
}

/*
  +-------------->CBR1---->CBR2---->CB-- +             +---->CBR3------->CBR4------>C2---+
  |                                      |             |                                 |
  |                                      V             |                                 V
Relu1---------------------------------->Add1-------->Relu2-------------->C1------------>Add2

Above pattern to be converted to the below one.

  +-------------->CBR1---->CBR2---->CB-- +             +---->CBR3------->CBR4------>C2---+
  |                                      |             |                                 |
  |                                      V             |                                 V
Relu1----------->Maxpool--------------->Add1-------->Relu2-------------->C1------------>Add2

Add2 is the concerned zn
C1 index = n1_index, with strides {2, 2}
C2 index = n2_index, with strides {1, 1}
CB index = an_p1ID/n3_index, with strides {1, 1}
CBR3 index = n2p index, with strides {2, 2}
Relu2 index = reluParentId/_reluParentId
Add1 index = addNodeId
Relu1 index = an_p2ID/rootReluParentID
CBR1 index = n4_index

After finding the pattern. Make following changes -
-Strides of C1 to {1, 1}
-Strides of CBR3 to {1, 1}
-Strides of CB to {2, 2}
-Add a Maxpool node with strides {2, 2} and kernel shape {1, 1}

Similar graph modification will happen in the Fusion case.
*/

void ZendnnGraphTransformer::optimizeForNonFusionCase(ZendnnSubgraph
        &subgraph) {
    /* This is to reduce the search space for pattern identifications.
    * There may be several subgraphs with #of nodes < 11
    * Non fusion of CONV_ADD case calls for pattern identification with-in a subgraph
    * with >=11 nodes.
    */

    if (subgraph.GetZendnnNodes().size() < 11) {
        return;
    }
    size_t index = 0;
    bool patternFound = true;
    while (patternFound) {
        auto tempZenNodes = subgraph.GetZendnnNodes();
        for (auto zn : tempZenNodes) {
            patternFound = false;
            index +=1;
            if (zn == nullptr) {
                continue;
            }
            if (!(zn->OpType() == "Add" && zn->InputCount() == 2)) {
                continue;
            }

            if (!(zn->Input(0).GetProducer().Exists() &&
                    zn->Input(1).GetProducer().Exists())) {
                continue;
            }

            auto n1_index = zn->Input(0).GetProducer().GetNode()->Index();
            auto n2_index = zn->Input(1).GetProducer().GetNode()->Index();

            if (!((subgraph.GetZendnnNode(n1_index)->OpType() == "Conv") &&
                    (subgraph.GetZendnnNode(n2_index)->OpType() == "Conv"))) {
                continue;
            }

            auto attr = subgraph.GetZendnnNode(n1_index)->Attributes().find("strides");
            std::vector<int64_t> n1strides;
            if (attr != subgraph.GetZendnnNode(n1_index)->Attributes().end()) {
                auto &proto = attr->second();
                Status status = GetStridesAttr(proto, n1strides);
            }
            else {
                continue;
            }

            attr = subgraph.GetZendnnNode(n2_index)->Attributes().find("strides");
            std::vector<int64_t> n2strides;
            if (attr != subgraph.GetZendnnNode(n2_index)->Attributes().end()) {
                auto &proto = attr->second();
                Status status = GetStridesAttr(proto, n2strides);
            }
            else {
                continue;
            }

            /* Either n1 or n2 should be 2,2 and 1,1 mutually. Else exit. */
            if (!((n1strides == std::vector<int64_t> {1, 1}) &&
                    (n2strides == std::vector<int64_t> {2, 2})) ||
                    ((n1strides == std::vector<int64_t> {2, 2}) &&
                     (n2strides == std::vector<int64_t> {1, 1}))) {
                continue;
            }

            /* Expecting node1 strides={2, 2} and node2 strides={1, 1}. Else swap for convention.*/
            if (n1strides == std::vector<int64_t> {1, 1}) {
                auto temp = n1_index;
                n1_index = n2_index;
                n2_index = temp;
            }

            if (!subgraph.GetZendnnNode(n1_index)->Input(0).GetProducer().Exists()) {
                continue;
            }

            auto reluParentID = subgraph.GetZendnnNode(n1_index)->Input(
                                    0).GetProducer().GetNode()->Index();

            if (subgraph.GetZendnnNode(reluParentID) == nullptr) {
                continue;
            }

            if (subgraph.GetZendnnNode(reluParentID)->OpType() != "Relu") {
                continue;
            }

            auto _reluParentID = n2_index;
            auto n2p = _reluParentID;

            if (subgraph.GetZendnnNode(_reluParentID) == nullptr ||
                    subgraph.GetZendnnNode(n2p) == nullptr) {
                continue;
            }

            /* According to the pattern, this node should be available at 3 nodes above. */
            for (int i = 0; i < 3; i++) {
                n2p = _reluParentID;
                if (subgraph.GetZendnnNode(_reluParentID)->Input(0).GetProducer().Exists()) {
                    _reluParentID = subgraph.GetZendnnNode(_reluParentID)->Input(
                                        0).GetProducer().GetNode()->Index();
                }
            }

            if (reluParentID != _reluParentID) {
                continue;
            }

            attr = subgraph.GetZendnnNode(n2p)->Attributes().find("strides");
            std::vector<int64_t> n2pstrides;
            if (attr != subgraph.GetZendnnNode(n2p)->Attributes().end()) {
                auto &proto = attr->second();
                Status status = GetStridesAttr(proto, n2pstrides);
            }
            else {
                continue;
            }

            if (n2pstrides != std::vector<int64_t> {2, 2}) {
                continue;
            }

            auto addNodeId = subgraph.GetZendnnNode(reluParentID)->Input(
                                 0).GetProducer().GetNode()->Index();

            if (subgraph.GetZendnnNode(addNodeId) == nullptr) {
                continue;
            }

            if (!((subgraph.GetZendnnNode(addNodeId)->OpType() == "Add") &&
                    (subgraph.GetZendnnNode(addNodeId)->InputCount() == 2))) {
                continue;
            }

            if (!(subgraph.GetZendnnNode(addNodeId)->Input(0).GetProducer().Exists() &&
                    subgraph.GetZendnnNode(addNodeId)->Input(1).GetProducer().Exists())) {
                continue;
            }

            auto an_p1ID = subgraph.GetZendnnNode(addNodeId)->Input(
                               0).GetProducer().GetNode()->Index();
            auto an_p2ID = subgraph.GetZendnnNode(addNodeId)->Input(
                               1).GetProducer().GetNode()->Index();

            /* Either an_p1ID or an_p2ID should be "Conv" and "Relu" mutually. Else exit. */
            if (!(((subgraph.GetZendnnNode(an_p1ID)->OpType() == "Conv") &&
                    (subgraph.GetZendnnNode(an_p2ID)->OpType() == "Relu")) ||
                    ((subgraph.GetZendnnNode(an_p1ID)->OpType() == "Relu") &&
                     (subgraph.GetZendnnNode(an_p2ID)->OpType() == "Conv")))) {
                continue;
            }

            /* Expecting an_p1ID = "Conv" and an_p2ID="Relu". Else swap for convention.*/
            if (subgraph.GetZendnnNode(an_p1ID)->OpType() == "Relu") {
                auto temp = an_p1ID;
                an_p1ID = an_p2ID;
                an_p2ID = temp;
            }

            attr = subgraph.GetZendnnNode(an_p1ID)->Attributes().find("strides");
            std::vector<int64_t> n3strides;
            if (attr != subgraph.GetZendnnNode(an_p1ID)->Attributes().end()) {
                auto &proto = attr->second();
                Status status = GetStridesAttr(proto, n3strides);
            }
            else {
                continue;
            }
            if (n3strides != std::vector<int64_t> {1, 1}) {
                continue;
            }

            auto rootReluParentID = an_p2ID;
            auto n3_index = an_p1ID;
            auto n4_index = n3_index;

            for (int i = 0; i < 2; i++) {
                if (subgraph.GetZendnnNode(n4_index)->Input(0).GetProducer().Exists()) {
                    n4_index = subgraph.GetZendnnNode(n4_index)->Input(
                                   0).GetProducer().GetNode()->Index();
                }
            }
            if (subgraph.GetZendnnNode(n4_index)->Input(0).GetProducer().Exists() &&
                    subgraph.GetZendnnNode(n4_index)->Input(0).GetProducer().GetNode()->Name() !=
                    subgraph.GetZendnnNode(rootReluParentID)->Name()) {
                continue;
            }

            /* If we have reached till here, we have found a pattern to insert MaxPool node.
            * Force n1_index node strides to {1, 1}, n2p index node strides
            * to {1, 1}, n3_index strides to {2, 2}.
            * Insert MaxPool node after rootReluParentID index and before addNodeId node (addNode).
            */

            subgraph.GetZendnnNode(n1_index)->SetStridesOpt(1);
            subgraph.GetZendnnNode(n2p)->SetStridesOpt(1);
            subgraph.GetZendnnNode(n3_index)->SetStridesOpt(2);

            auto dummyMaxPool = std::make_unique<ZendnnNode>();
            dummyMaxPool->Name() = "MaxPool_" + std::to_string(index);
            dummyMaxPool->OpType() = "MaxPool";
            dummyMaxPool->SetStridesOpt(2);

            auto dummyMaxPool_output_tensor_name = "DMaxPool_" + std::to_string(
                    index) + "_Y";
            auto dummyMaxPool_output_tensor = std::make_unique<ZendnnTensor>
                                              (dummyMaxPool_output_tensor_name);

            subgraph.InsertNode(subgraph.GetZendnnNode(rootReluParentID),
                                subgraph.GetZendnnNode(addNodeId), std::move(dummyMaxPool),
                                std::move(dummyMaxPool_output_tensor));

            patternFound = true;
            break;
        }
    }
}

void ZendnnGraphTransformer::ConvClip(ZendnnSubgraph &subgraph) {
    //global index of convclip
    static int conv_clip_index = 0;

    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);

        //look for conv clip pattern
        if (zendnn_node == nullptr) {
            continue;
        }

        if (zendnn_node->OpType() != "Conv") {
            continue;
        }

        if (zendnn_node->OutputCount() != 1) {
            ORT_THROW("Invalid Conv node");
        }

        if (zendnn_node->Output(0).Exists() &&
                zendnn_node->Output(0).GetConsumers().size() != 1) {
            continue;
        }

        //check whether conv node's only output tensor is outputting the subgraph
        {
            auto graph_outputs = subgraph.GetZendnnOutputs();
            if (std::find(graph_outputs.cbegin(), graph_outputs.cend(),
                          &zendnn_node->Output(0)) != graph_outputs.cend()) {
                continue;
            }
        }

        auto next_zendnn_node = zendnn_node->Output(0).GetConsumers()[0].GetNode();
        if (next_zendnn_node == nullptr) {
            continue;
        }
        if (next_zendnn_node->OpType() != "Clip") {
            continue;
        }

        //construct new node
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = zendnn_node->Name() + "_ConvClip_" + std::to_string(
                               conv_clip_index++);
        new_node->OpType() = zendnn_node->OpType() + "Clip";
        for (auto def : zendnn_node->Inputs()) {
            new_node->Inputs().push_back(def);
        }
        for (auto def : next_zendnn_node->Inputs()) {
            if (def->Exists() && def->GetProducer().Exists() &&
                    def->GetProducer().GetNode()->OpType() == "Conv") {
                continue;
            }

            new_node->Inputs().push_back(def);
        }
        for (auto def : next_zendnn_node->Outputs()) {
            new_node->Outputs().push_back(def);
        }
        new_node->Attributes().insert(zendnn_node->Attributes());
        new_node->Attributes().insert(next_zendnn_node->Attributes());

        //insert new node, remove original nodes, connect new edges
        LOGS_DEFAULT(INFO) << "fuse [" << zendnn_node->Name() << "] and [" <<
                           next_zendnn_node->Name() << "] into ConvClip";
        ResolveFusion(subgraph, {zendnn_node->Index(), next_zendnn_node->Index()},
                      std::move(new_node));
    }
}

void ZendnnGraphTransformer::BatchnormRelu(ZendnnSubgraph &subgraph) {
    static int bn_relu_index = 0;

    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();
    for (size_t index = 0; index < max_index; index++) {
        auto zendnn_node = subgraph.GetZendnnNode(index);

        //look for bn relu pattern
        if (zendnn_node == nullptr) {
            continue;
        }

        if (zendnn_node->OpType() != "BatchNormalization") {
            continue;
        }

        if (zendnn_node->OutputCount() != 1) {
            ORT_THROW("Invalid BN node");
        }

        if (zendnn_node->Output(0).Exists() &&
                zendnn_node->Output(0).GetConsumers().size() != 1) {
            continue;
        }

        //check whether bn node's only output tensor is outputting the subgraph
        {
            auto graph_outputs = subgraph.GetZendnnOutputs();
            if (std::find(graph_outputs.cbegin(), graph_outputs.cend(),
                          &zendnn_node->Output(0)) != graph_outputs.cend()) {
                continue;
            }
        }

        auto next_zendnn_node = zendnn_node->Output(0).GetConsumers()[0].GetNode();
        if (next_zendnn_node == nullptr) {
            continue;
        }
        if (next_zendnn_node->OpType() != "Relu") {
            continue;
        }

        //construct new node
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = zendnn_node->Name() + "_BNRelu_" + std::to_string(
                               bn_relu_index++);
        new_node->OpType() = "BatchnormRelu";
        for (auto def : zendnn_node->Inputs()) {
            new_node->Inputs().push_back(def);
        }
        for (auto def : next_zendnn_node->Outputs()) {
            new_node->Outputs().push_back(def);
        }
        new_node->Attributes().insert(zendnn_node->Attributes());

        //insert new node, remove original nodes, connect new edges
        LOGS_DEFAULT(INFO) << "fuse [" << zendnn_node->Name() << "] and [" <<
                           next_zendnn_node->Name() << "] into BatchnormRelu";
        ResolveFusion(subgraph, {zendnn_node->Index(), next_zendnn_node->Index()},
                      std::move(new_node));
    }
}

void ZendnnGraphTransformer::FuseLN(ZendnnSubgraph &subgraph) {
    //global index of ln_block
    static int ln_block = 0;
    //traverse with max index as there will be empty nodes due to fusion
    size_t max_index = subgraph.GetMaxNodeIndex();

    std::vector<size_t> remove_indices;

    for (size_t index = 0; index < max_index; index++) {
        remove_indices.clear();
        auto zendnn_node = subgraph.GetZendnnNode(index);
        /* Look for layernormalization pattern, like below
         * [X]-->(ReduceMean)     (Pow)-->(ReduceMean)-->(Add)-->(Sqrt)-->(Div)-->(Mul)-->(Add)-->[Y]
         *     |      |             ^                                       ^
         *     |      v             |                                       |
         *     |--->(Sub)---------->|-------------------------------------->|
         */
        if (zendnn_node == nullptr) {
            continue;
        }

        if (zendnn_node->OpType() != "ReduceMean") {
            continue;
        }

        remove_indices.push_back(zendnn_node->Index());

        //construct new node
        auto new_node = std::make_unique<ZendnnNode>();
        new_node->Name() = zendnn_node->Name() + "_LN_" + std::to_string(ln_block++);
        new_node->OpType() = "LayerNormalization";

        /* Push Reducemean inputs to LayerNormalization node. */
        for (auto def : zendnn_node->Inputs()) {
            new_node->Inputs().push_back(def);
        }
        new_node->Attributes().insert(zendnn_node->Attributes());

        ZendnnNode *z_node = zendnn_node;
        auto next_node = z_node->Output(0).GetConsumers()[0].GetNode();
        if (next_node->OpType() != "Sub") {
            continue;
        }

        remove_indices.push_back(next_node->Index());

        z_node = next_node;

        ZendnnNode *_next_node[2];
        _next_node[0] = z_node->Output(0).GetConsumers()[0].GetNode();
        if (_next_node[0]->OpType() != "Pow" && _next_node[0]->OpType() != "Div") {
            continue;
        }

        /* Making a convention. Index 0=Pow, 1=Div */
        if (_next_node[0]->OpType() == "Pow") {
            _next_node[1] = z_node->Output(0).GetConsumers()[1].GetNode();
        }
        else {
            _next_node[0] = z_node->Output(0).GetConsumers()[1].GetNode();
            _next_node[1] = z_node->Output(0).GetConsumers()[0].GetNode();
        }

        if (_next_node[1]->OpType() != "Div") {
            continue;
        }

        remove_indices.push_back(_next_node[0]->Index());

        auto rm_node = _next_node[0]->Output(0).GetConsumers()[0].GetNode();
        if (rm_node->OpType() != "ReduceMean") {
            continue;
        }
        new_node->Attributes().insert(rm_node->Attributes());

        remove_indices.push_back(rm_node->Index());

        auto epsAdd_node = rm_node->Output(0).GetConsumers()[0].GetNode();
        if (epsAdd_node->OpType() != "Add") {
            continue;
        }

        remove_indices.push_back(epsAdd_node->Index());

        auto sqrt_node = epsAdd_node->Output(0).GetConsumers()[0].GetNode();
        if (sqrt_node->OpType() != "Sqrt") {
            continue;
        }

        remove_indices.push_back(sqrt_node->Index());

        auto div_node = sqrt_node->Output(0).GetConsumers()[0].GetNode();
        if (div_node->OpType() != "Div") {
            continue;
        }

        remove_indices.push_back(div_node->Index());

        auto mul_scale_node = div_node->Output(0).GetConsumers()[0].GetNode();
        if (mul_scale_node->OpType() != "Mul") {
            continue;
        }
        /* Push scale inputs to LayerNormalization node. */
        for (auto def : mul_scale_node->Inputs()) {
            if (def->GetProducer().Exists() &&
                    def->GetProducer().GetNode()->OpType() == "Div") {
                continue;
            }
            new_node->Inputs().push_back(def);
        }

        remove_indices.push_back(mul_scale_node->Index());

        auto add_shift_node = mul_scale_node->Output(0).GetConsumers()[0].GetNode();
        if (add_shift_node->OpType() != "Add") {
            continue;
        }
        for (auto def : add_shift_node->Inputs()) {
            if (def->GetProducer().Exists() &&
                    def->GetProducer().GetNode()->OpType() == "Mul") {
                continue;
            }
            new_node->Inputs().push_back(def);
        }
        /* Push shift Outputs to LayerNormalization node. */
        for (auto def : add_shift_node->Outputs()) {
            new_node->Outputs().push_back(def);
        }

        remove_indices.push_back(add_shift_node->Index());
        /*
            for (auto e : remove_indices) {
              LOGS_DEFAULT(INFO) <<"Removing : | " <<e;
            }
        */
        //LOGS_DEFAULT(ERROR) <<"Fusing block to LayerNormalization node ";
        ResolveFusion(subgraph, remove_indices, std::move(new_node));
    }
}


}  // namespace ort_zendnn
}  // namespace onnxruntime
