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

#include <vector>
#include <string>
#include <map>
#include <limits>
#include "zendnn.hpp"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_zendnn {

class ZendnnNode;

class ZendnnNodeArg {
  public:
    ZendnnNodeArg(ZendnnNode *node, size_t index, bool is_output)
        : node_(node), index_(index), is_output_(is_output) {};
    ZendnnNodeArg() = default;
    ZendnnNode *GetNode() {
        return node_;
    };
    size_t GetIndex() {
        return index_;
    };
    bool IsOutput() {
        return is_output_;
    };
    bool Exists() {
        return node_ != nullptr;
    };
    bool operator==(const ZendnnNodeArg &e) const {
        return node_ == e.node_ && index_ == e.index_ && is_output_ == e.is_output_;
    };

  private:
    ZendnnNode *node_ = nullptr;
    size_t index_ = std::numeric_limits<size_t>::max();
    bool is_output_ = false;
};

class ZendnnTensor {
  public:
    ZendnnTensor(const NodeArg *arg, bool isConstantInitializer = false);
    ZendnnTensor(std::string name);
    ZendnnTensor() = default;
    std::string Name() const;
    zendnn::memory::dims Dim() const;
    zendnn::memory::data_type Type() const;
    zendnn::memory::format_tag Format();
    // Check whether the tensor is dynamic, e.g. contains unspecified dimension
    bool IsDynamic();
    // Check whether the tensor is constant initializer
    bool IsConstant();
    // Check whether the tensor exsits for optional input output
    bool Exists();
    std::vector<ZendnnNodeArg> &GetConsumers() {
        return consumers_;
    };
    size_t GetConsumersCount() {
        return consumers_.size();
    };
    ZendnnNodeArg &GetProducer() {
        return producer_;
    };
    void SetProducer(const ZendnnNodeArg &arg);
    void ResetProducer();
    void AddConsumer(const ZendnnNodeArg &arg);
    void RemoveConsumer(const ZendnnNodeArg &arg);

  private:

    const ONNX_NAMESPACE::TensorShapeProto *GetShape() const;

    std::string tensor_name_;
    ONNX_NAMESPACE::DataType arg_type_;
    std::unique_ptr<ONNX_NAMESPACE::TypeProto> arg_type_proto_;
    //a tensor can have no producer (input.initializer) or no consumer (output for subgraph)
    ZendnnNodeArg producer_;
    std::vector<ZendnnNodeArg> consumers_;
    bool isConstant_;
};

class ZendnnNode {
  public:
    ZendnnNode(const Node *node);
    ZendnnNode() = default;
    std::string &Name();
    size_t &Index();
    std::string &OpType();
    ZendnnTensor &Input(int index);
    size_t InputCount();
    ZendnnTensor &Output(int index);
    size_t OutputCount();
    NodeAttributes &Attributes();
    std::vector<ZendnnTensor *> &Inputs();
    std::vector<ZendnnTensor *> &Outputs();
    int SinceVersion();
    void AppendPostOp(std::string op);
    const std::vector<std::string> &GetPostOps();
    /* Essential functions for performing inplace memory operations;
     * like inplace concatenation for several concat patterns
     */
    bool isInplaceMemoryNode = false;
    void setOpType(std::string op_type);
    void SetStridesOpt(int value);
    int GetStridesOpt();
    zendnn::memory::data_type Type(std::string data_type);

  private:
    int since_version_;
    std::vector<ZendnnTensor *> inputs_;
    std::vector<ZendnnTensor *> outputs_;
    static ZendnnTensor empty_tensor_;
    std::string name_;  // node can have empty/duplicate name, rely on index instead
    std::string op_type_;
    size_t index_ = std::numeric_limits<size_t>::max();
    std::unique_ptr<NodeAttributes> attr_ = NodeAttributes::Create();
    std::vector<std::string> postops_;
    /* The below flag is for strides trick optimization.
    * It's used only in CONV and POOL nodes.
    * force strides. 0=(Retain from ORT); 1=(1, 1); 2=(2, 2).
    * This flag is for strides trick optimization in resnet variant models.
    */
    int strides_opt = 0;
};

class ZendnnSubgraph {
  public:
    ZendnnSubgraph(const GraphViewer &graph_viewer);
    std::vector<ZendnnNode *> GetZendnnNodes();
    ZendnnNode *GetZendnnNode(size_t node_index);
    ZendnnTensor *GetZendnnTensor(const std::string &tensor_name);
    size_t GetMaxNodeIndex();
    std::vector<size_t> GetZendnnNodesInTopologicalOrder();
    std::vector<ZendnnTensor *> GetZendnnInputs();
    std::vector<ZendnnTensor *> GetZendnnOutputs();
    std::vector<ZendnnTensor *> GetZendnnInitializers();
    // build the subgraph IR
    void Build(const GraphViewer &graph_viewer);
    //check whether the subgraph is dynamic
    void TopoSort();
    bool IsDynamic();
    void AddNode(std::unique_ptr<ZendnnNode> new_node);
    void RemoveNode(size_t node_index);
    void AddTensor(std::unique_ptr<ZendnnTensor> new_tensor);
    void RemoveTensor(const std::string &tensor_name);
    void InsertNode(ZendnnNode *node1, ZendnnNode *node2,
                    std::unique_ptr<ZendnnNode> newNode,
                    std::unique_ptr<ZendnnTensor> newNodeOutput);

  private:
    //graph owns all nodes
    std::vector<std::unique_ptr<ZendnnNode>> zendnn_nodes_;
    std::vector<size_t> nodes_in_topological_order_;
    //graph owns all tensors
    std::unordered_map<std::string, std::unique_ptr<ZendnnTensor>> zendnn_tensors_;
    std::vector<ZendnnTensor *> inputs_;
    std::vector<ZendnnTensor *>
    outputs_; //output should never get deleted from graph transformation
    std::vector<ZendnnTensor *> initializers_;
    bool is_dynamic_;
};
}  // namespace ort_zendnn
}  // namespace onnxruntime
