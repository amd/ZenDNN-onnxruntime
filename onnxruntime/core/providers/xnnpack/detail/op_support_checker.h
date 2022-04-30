// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>

namespace onnxruntime {
class GraphViewer;
class Node;

namespace xnnpack {
class NodeSupportChecker {
 public:
  NodeSupportChecker(const GraphViewer& graph,
                     const std::unordered_set<const Node*>& supported_nodes)
      : graph_{graph},
        supported_nodes_{supported_nodes} {
  }

  bool IsNodeSupported(const Node& node, bool matched_kernel);

 private:
  const GraphViewer& graph_;
  const std::unordered_set<const Node*>& supported_nodes_;  // previously selected nodes
};

}  // namespace xnnpack
}  // namespace onnxruntime
