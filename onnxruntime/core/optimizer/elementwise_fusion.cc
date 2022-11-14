// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/elementwise_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

namespace {

bool IsSameShapes(const TensorShapeProto* shape1, const TensorShapeProto* shape2) {
  if (!shape1 || !shape2) {
    return false;
  }
  if (shape1->dim_size() != shape2->dim_size()) {
    return false;
  }
  for (int i = 0; i < shape1->dim_size(); i++) {
    if (shape1->dim(i) != shape2->dim(i)) {
      return false;
    }
  }
  return true;
}

bool IsScalarShape(int rank, const TensorShapeProto* shape) {
  if (!shape) {
    return false;
  }
  if (shape->dim_size() > rank) {
    return false;
  }
  for (int i = 0; i < shape->dim_size(); i++) {
    if (!shape->dim(i).has_dim_value() || shape->dim(i).dim_value() != 1) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool ElementwiseFusion::IsSupportedNode(const Node& node) const {
  return (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7, 13, 14}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Sub", {7, 13, 14}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Mul", {7, 13, 14}) ||
          graph_utils::IsSupportedOptypeVersionAndDomain(node, "Div", {7, 13, 14})) &&
         graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders());
}

Status ElementwiseFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr) continue;  // node was removed

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!IsSupportedNode(node) || node.GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto* input1_shape = node.InputDefs()[0]->Shape();
    if (!input1_shape) {
      continue;
    }
    int rank = input1_shape->dim_size();
    const auto* input2_shape = node.InputDefs()[1]->Shape();
    if (!IsSameShapes(input1_shape, input2_shape) && !IsScalarShape(rank, input2_shape)) {
      continue;
    }

    Node& second_node = *graph.GetNode(node.OutputNodesBegin()->Index());
    if (!IsSupportedNode(second_node)) {
      continue;
    }

    const auto* input3_shape = second_node.InputDefs()[1]->Shape();
    if (!IsSameShapes(input1_shape, input3_shape) && !IsScalarShape(rank, input3_shape)) {
      continue;
    }

    std::vector<std::string> op_types{node.OpType(), second_node.OpType()};
    InlinedVector<std::reference_wrapper<Node>> nodes_to_fuse{node, second_node};
    if (second_node.GetOutputEdgesCount() == 1) {
      Node& third_node = *graph.GetNode(second_node.OutputNodesBegin()->Index());
      if (IsSupportedNode(third_node)) {
        const auto* input4_shape = third_node.InputDefs()[1]->Shape();
        if (IsSameShapes(input1_shape, input4_shape) || IsScalarShape(rank, input4_shape)) {
          op_types.push_back(third_node.OpType());
          nodes_to_fuse.push_back(third_node);
        }
      }
    }

    InlinedVector<NodeArg*> input_args{nodes_to_fuse[0].get().MutableInputDefs()[0]};
    for (size_t i = 0; i < nodes_to_fuse.size(); ++i) {
      input_args.emplace_back(nodes_to_fuse[i].get().MutableInputDefs()[1]);
    }
    Node& fused_node = graph.AddNode(
        graph.GenerateNodeName("FusedElementwise"), "FusedElementwise", "Fused multiple elementwise nodes", input_args,
        {nodes_to_fuse[nodes_to_fuse.size() - 1].get().MutableOutputDefs()[0]}, {}, kMSDomain);
    fused_node.AddAttribute("op_types", op_types);
    fused_node.SetExecutionProviderType(node.GetExecutionProviderType());

    for (Node& n : nodes_to_fuse) {
      graph_utils::RemoveNodeOutputEdges(graph, n);
      graph.RemoveNode(n.Index());
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
