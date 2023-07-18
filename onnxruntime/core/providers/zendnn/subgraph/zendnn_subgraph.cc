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

#include "zendnn_subgraph.h"
#include <queue>

namespace onnxruntime {
namespace ort_zendnn {

ZendnnTensor ZendnnNode::empty_tensor_ = ZendnnTensor("");

ZendnnTensor::ZendnnTensor(const NodeArg *arg, bool isConstantInitializer) {
    if (!arg || !arg->Exists()) {
        tensor_name_ = "";
    }
    else {
        tensor_name_ = arg->Name();
    }
    // because the passed in ort graph will be released after compile
    // need to save the type/shape in zendnn IR
    arg_type_ = arg->Type();
    arg_type_proto_ = ONNX_NAMESPACE::TypeProto::Create();
    arg_type_proto_->copy_from(arg->TypeAsProto());
    isConstant_ = isConstantInitializer;
}

ZendnnTensor::ZendnnTensor(std::string name) {
    tensor_name_ = name;
    arg_type_ = nullptr;
    arg_type_proto_ = nullptr;
}

std::string ZendnnTensor::Name() const {
    return tensor_name_;
}

const ONNX_NAMESPACE::TensorShapeProto *ZendnnTensor::GetShape() const {
    if (arg_type_proto_ == nullptr || arg_type_ == nullptr) {
        return nullptr;
    }

    if (arg_type_proto_->value_case() !=
            ONNX_NAMESPACE::TypeProto::ValueCase::kTensorType) {
        return nullptr;
    }
    auto &tensor_type = arg_type_proto_->tensor_type();
    if (tensor_type.has_shape()) {
        return &tensor_type.shape();
    }
    return nullptr;
}

zendnn::memory::dims ZendnnTensor::Dim() const {
    if (arg_type_proto_ == nullptr || arg_type_ == nullptr) {
        return zendnn::memory::dims();
    }
    auto *shape_proto = GetShape();
    // a shape without any information
    if (shape_proto == nullptr) {
        LOGS_DEFAULT(INFO) << "nullptr shape for " << arg_type_ << ": " << tensor_name_;
        return zendnn::memory::dims();
    }
    std::vector<int64_t> shape;
    const auto &dims = shape_proto->dim();
    for (const auto &dim : dims) {
        bool has_dim_value = dim.value_case() == dim.kDimValue;
        if (!has_dim_value) {
            LOGS_DEFAULT(INFO) << "Dynamic shape for " << arg_type_ << ": " << tensor_name_;
            shape.push_back(ZENDNN_RUNTIME_DIM_VAL);
        }
        else {
            shape.push_back(dim.dim_value());
        }
    }
    //make scaler as having dimension of 1
    if (shape.size() == 0) {
        shape.push_back(1);
    }
    auto zendnn_dims = zendnn::memory::dims(shape);
    return zendnn_dims;
}

zendnn::memory::data_type ZendnnNode::Type(std::string data_type) {
    if (data_type == "DT_FLOAT") {
        return zendnn::memory::data_type::f32;
    }
    else if (data_type == "DT_QUINT8") {
        return zendnn::memory::data_type::u8;
    }
    else if (data_type == "DT_INT32") {
        return zendnn::memory::data_type::s32;
    }
    else if (data_type == "DT_QINT8") {
        return zendnn::memory::data_type::s8;
    }
    ORT_THROW("Unsupported data type: ", data_type);
}

zendnn::memory::data_type ZendnnTensor::Type() const {
    if (arg_type_proto_ == nullptr) {
        ORT_THROW("Invoke ZendnnTensor's arg_type_proto_ not initialized yet.");
    }
    auto data_type = arg_type_proto_->tensor_type().elem_type();
    switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
        return zendnn::memory::data_type::undef;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        return zendnn::memory::data_type::f16;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
        return zendnn::memory::data_type::bf16;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        return zendnn::memory::data_type::f32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        // ZenDNN does not have support for tensors of int64_t so we just say
        // the tensor is int32_t and then use casting in the actual operator
        // to convert the zendnn::memory::data_handle to an int64_t*.  Care
        // must be taken that an int64_t tensor does not make it pass the
        // node capability check unless the operator is explicitly expecting
        // the int64_t
        return zendnn::memory::data_type::s32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        return zendnn::memory::data_type::s32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        return zendnn::memory::data_type::s8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        return zendnn::memory::data_type::u8;
    // Same here, we use u8 as the handler for bool
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
        return zendnn::memory::data_type::u8;
    default:
        ORT_THROW("Unsupported data type: ", data_type);
    }
}

bool ZendnnTensor::IsDynamic() {
    if (Dim().size() == 0) {
        return true;
    }
    for (auto dim : Dim()) {
        if (dim == ZENDNN_RUNTIME_DIM_VAL) {
            return true;
        }
    }
    return false;
}

bool ZendnnTensor::IsConstant() {
    return isConstant_;
}

bool ZendnnTensor::Exists() {
    return !(tensor_name_ == "");
}

zendnn::memory::format_tag ZendnnTensor::Format() {
    return zendnn::memory::format_tag::any;
}

void ZendnnTensor::SetProducer(const ZendnnNodeArg &arg) {
    producer_ = arg;
}

void ZendnnTensor::ResetProducer() {
    producer_ = ZendnnNodeArg();
}

void ZendnnTensor::AddConsumer(const ZendnnNodeArg &arg) {
    consumers_.push_back(arg);
}

void ZendnnTensor::RemoveConsumer(const ZendnnNodeArg &arg) {
    consumers_.erase(std::remove(consumers_.begin(), consumers_.end(), arg),
                     consumers_.end());
}

ZendnnNode::ZendnnNode(const Node *node) {
    since_version_ = node->SinceVersion();
    name_ = node->Name();
    op_type_ = node->OpType();
    attr_->insert(node->GetAttributes());
}

std::string &ZendnnNode::Name() {
    return name_;
}

std::string &ZendnnNode::OpType() {
    return op_type_;
}

void ZendnnNode::SetStridesOpt(int value) {
    strides_opt = value;
}

int ZendnnNode::GetStridesOpt() {
    return strides_opt;
}

std::vector<ZendnnTensor *> &ZendnnNode::Inputs() {
    return inputs_;
}

std::vector<ZendnnTensor *> &ZendnnNode::Outputs() {
    return outputs_;
}

size_t &ZendnnNode::Index() {
    return index_;
}

ZendnnTensor &ZendnnNode::Input(int index) {
    if (inputs_.size() <= (size_t)index) {
        return empty_tensor_;
    }
    if (inputs_[index] && inputs_[index]->Exists()) {
        return *inputs_[index];
    }
    return empty_tensor_;
}

size_t ZendnnNode::InputCount() {
    return inputs_.size();
}

ZendnnTensor &ZendnnNode::Output(int index) {
    return *outputs_[index];
}

size_t ZendnnNode::OutputCount() {
    return outputs_.size();
}

NodeAttributes &ZendnnNode::Attributes() {
    return *attr_;
}

int ZendnnNode::SinceVersion() {
    return since_version_;
}

void ZendnnNode::AppendPostOp(std::string op) {
    postops_.push_back(op);
}

const std::vector<std::string> &ZendnnNode::GetPostOps() {
    return postops_;
}

void ZendnnNode :: setOpType(std::string op_type) {
    op_type_ = op_type;
}

ZendnnSubgraph::ZendnnSubgraph(const GraphViewer &graph_viewer) {
    Build(graph_viewer);
    is_dynamic_ = false;
    for (auto input : GetZendnnInputs()) {
        if (input->IsDynamic()) {
            is_dynamic_ = true;
            break;
        }
    }
}

bool ZendnnSubgraph::IsDynamic() {
    return is_dynamic_;
}

void ZendnnSubgraph::TopoSort() {
    nodes_in_topological_order_.clear();

    std::unordered_map<size_t, int> indegrees;
    for (auto &node : zendnn_nodes_) {
        if (node.get()) {
            indegrees[node->Index()] = 0;
        }
    }

    for (auto &e : zendnn_tensors_) {
        auto tensor = e.second.get();
        if (tensor->Exists() && tensor->GetProducer().GetNode()) {
            for (auto edge : tensor->GetConsumers()) {
                if (edge.GetNode()) {
                    indegrees[edge.GetNode()->Index()]++;
                }
            }
        }
    }

    std::queue<ZendnnNode *> queue;
    for (auto e : indegrees) {
        if (e.second == 0) {
            queue.push(zendnn_nodes_[e.first].get());
        }
    }

    //need to make sure all indegrees are computed before doing bfs
    while (!queue.empty()) {
        auto cur = queue.front();
        queue.pop();
        nodes_in_topological_order_.push_back(cur->Index());
        for (auto output : cur->Outputs()) {
            if (output && output->Exists()) {
                for (auto edge : output->GetConsumers()) {
                    indegrees[edge.GetNode()->Index()] -= 1;
                    if (indegrees[edge.GetNode()->Index()] == 0) {
                        queue.push(edge.GetNode());
                    }
                }
            }
        }
    }
    assert(indegrees.size() == nodes_in_topological_order_.size());
}

std::vector<size_t> ZendnnSubgraph::GetZendnnNodesInTopologicalOrder() {
    TopoSort();
    return nodes_in_topological_order_;
}

ZendnnNode *ZendnnSubgraph::GetZendnnNode(size_t node_index) {
    return zendnn_nodes_[node_index].get();
}

ZendnnTensor *ZendnnSubgraph::GetZendnnTensor(const std::string &tensor_name) {
    if (zendnn_tensors_.count(tensor_name)) {
        return zendnn_tensors_[tensor_name].get();
    }
    else {
        return nullptr;
    }
}


size_t ZendnnSubgraph::GetMaxNodeIndex() {
    return zendnn_nodes_.size();
}

std::vector<ZendnnNode *> ZendnnSubgraph::GetZendnnNodes() {
    std::vector<ZendnnNode *> result;
    for (auto &node : zendnn_nodes_) {
        if (node.get()) {
            result.push_back(node.get());
        }
    }
    return result;
}

std::vector<ZendnnTensor *> ZendnnSubgraph::GetZendnnInputs() {
    return inputs_;
}

std::vector<ZendnnTensor *> ZendnnSubgraph::GetZendnnOutputs() {
    return outputs_;
}

std::vector<ZendnnTensor *> ZendnnSubgraph::GetZendnnInitializers() {
    return initializers_;
}

void ZendnnSubgraph::RemoveNode(size_t node_index) {
    zendnn_nodes_[node_index].reset(nullptr);
}

void ZendnnSubgraph::RemoveTensor(const std::string &tensor_name) {
    inputs_.erase(std::remove_if(inputs_.begin(),
    inputs_.end(), [=](ZendnnTensor* t) {
        return t->Name() == tensor_name;
    }), inputs_.end());
    initializers_.erase(std::remove_if(initializers_.begin(),
    initializers_.end(), [=](ZendnnTensor* t) {
        return t->Name() == tensor_name;
    }), initializers_.end());
    zendnn_tensors_.erase(tensor_name);
}

void ZendnnSubgraph::AddTensor(std::unique_ptr<ZendnnTensor> new_tensor) {
    if (!zendnn_tensors_.count(new_tensor->Name())) {
        zendnn_tensors_.emplace(new_tensor->Name(), std::move(new_tensor));
    }
    else {
        ORT_THROW("tensor exists, modify or delete first before inseting");
    }
}

void ZendnnSubgraph::AddNode(std::unique_ptr<ZendnnNode> new_node) {
    auto index = zendnn_nodes_.size();
    zendnn_nodes_.emplace_back(std::move(new_node));
    zendnn_nodes_.back()->Index() = index;
}

/*
        tensor_to_redirect
node1---------------------->node2

will be converted to

        tensor_to_redirect            newNodeOutput
node1---------------------->newNode-------------------->node2
*/

void ZendnnSubgraph::InsertNode(ZendnnNode *node1, ZendnnNode *node2,
                                std::unique_ptr<ZendnnNode> newNode,
                                std::unique_ptr<ZendnnTensor> newNodeOutput) {

    auto tensor_to_redirect = GetZendnnTensor(node1->Output(0).Name());

    //node2_consumer_index tells what is the consumer index of node2 for node1
    size_t node2_consumer_index = (size_t)-1;
    for (size_t k = 0; k < tensor_to_redirect->GetConsumers().size(); k++) {
        if (tensor_to_redirect->GetConsumers()[k].GetNode() == node2) {
            node2_consumer_index = tensor_to_redirect->GetConsumers()[k].GetIndex();
        }
    }
    if (static_cast<int>(node2_consumer_index) == -1) {
        return;
    }

    tensor_to_redirect->RemoveConsumer(ZendnnNodeArg(node2, node2_consumer_index,
                                       false));
    tensor_to_redirect->AddConsumer(ZendnnNodeArg(newNode.get(),
                                    node2_consumer_index, false));

    // node2 can have > 1 nodes as Inputs, node1_producer_index tells which input of node2 is node1
    size_t node1_producer_index = (size_t)-1;
    for (size_t k = 0; k < node2->InputCount(); k++) {
        if (node2->Input((int)k).GetProducer().GetNode() == node1) {
            node1_producer_index = k;
        }
    }
    if (static_cast<int>(node1_producer_index) == -1) {
        return;
    }

    //remove the input node1 from node2's input list.
    node2->Inputs().erase(node2->Inputs().begin() + node1_producer_index);

    newNodeOutput->ResetProducer();
    //index 0 is passed as the index of newNode as newNode has only 1 consumer and producer each.
    newNodeOutput->SetProducer(ZendnnNodeArg(newNode.get(), 0, true));
    newNodeOutput->AddConsumer(ZendnnNodeArg(node2, 0, false));

    newNode->Outputs().push_back(newNodeOutput.get());
    newNode->Inputs().push_back(tensor_to_redirect);

    //add the input newNode to the node2's input list.
    node2->Inputs().insert(node2->Inputs().begin() + node1_producer_index,
                           newNodeOutput.get());

    AddTensor(std::move(newNodeOutput));
    AddNode(std::move(newNode));
}

void ZendnnSubgraph::Build(const GraphViewer &graph_viewer) {
    //establish nodes, tensors and nodeargs
    const auto &node_indices = graph_viewer.GetNodesInTopologicalOrder();
    for (size_t i = 0; i < node_indices.size(); i++) {
        const auto *node(graph_viewer.GetNode(node_indices[i]));
        AddNode(std::make_unique<ZendnnNode>(node));
        auto zendnn_node = zendnn_nodes_.back().get();
        std::vector<ZendnnTensor *> inputs;
        size_t index = 0;
        for (auto input : node->InputDefs()) {
            if (input && input->Exists() && input->Name() != "") {
                if (!zendnn_tensors_.count(input->Name())) {
                    zendnn_tensors_[input->Name()] =
                        std::make_unique<ZendnnTensor>(input,
                                                       graph_viewer.IsConstantInitializer(input->Name(), true));
                }
                zendnn_tensors_[input->Name()]->AddConsumer(ZendnnNodeArg(zendnn_node, index,
                        false));
                inputs.push_back(zendnn_tensors_[input->Name()].get());
            }
            else {
                inputs.push_back(nullptr);
            }
            index++;
        }
        std::vector<ZendnnTensor *> outputs;
        index = 0;
        for (auto output : node->OutputDefs()) {
            if (output && output->Exists() && output->Name() != "") {
                if (!zendnn_tensors_.count(output->Name())) {
                    zendnn_tensors_[output->Name()] = std::make_unique<ZendnnTensor>(output);
                }
                zendnn_tensors_[output->Name()]->SetProducer(ZendnnNodeArg(zendnn_node, index,
                        true));
                outputs.push_back(zendnn_tensors_[output->Name()].get());
            }
            else {
                outputs.push_back(nullptr);
            }
            index++;
        }
        zendnn_node->Inputs() = inputs;
        zendnn_node->Outputs() = outputs;
    }

    //all tensors should have been established in graph
    //establish inputs, outputs and initializers
    //graph inputs including initializers and outputs can be deleted by graph transformation (eg, gelu fusion)
    //delete unneeded inputs don't affect onnxruntime passing them as input data handle
    //delete unneeded outputs will cause ep to output to fewer data handles then expected
    for (const auto *node_arg : graph_viewer.GetInputsIncludingInitializers()) {
        inputs_.push_back(zendnn_tensors_[node_arg->Name()].get());
    }

    for (const auto *node_arg : graph_viewer.GetOutputs()) {
        outputs_.push_back(zendnn_tensors_[node_arg->Name()].get());
    }

    for (auto &initializer : graph_viewer.GetAllInitializedTensors()) {
        auto &name = initializer.first;
        initializers_.push_back(zendnn_tensors_[name].get());
    }
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
