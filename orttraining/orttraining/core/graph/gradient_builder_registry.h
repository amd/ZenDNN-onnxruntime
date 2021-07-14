// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <functional>
#include "gradient_builder_base.h"
#include "generic_registry.h"

namespace onnxruntime {
namespace training {

typedef GenericRegistry<GradientBuilderBase, const GradientGraphConfiguration&,
                        Graph*&,                                 // graph
                        const Node*&,                            // node
                        const std::unordered_set<std::string>&,  // gradient_inputs
                        const std::unordered_set<std::string>&,  // gradient_outputs
                        const logging::Logger&>
    GradientRegistryType;

class GradientBuilderRegistry : public GradientRegistryType {
 public:
  void RegisterGradientBuilders();

  static GradientBuilderRegistry& GetInstance() {
    static GradientBuilderRegistry instance;
    return instance;
  }

  template <typename DerivedType>
  void RegisterGradientBuilder(const std::string& op_name, const std::string& name_with_ver_range) {
    auto range = op_to_ver_range_map_.equal_range(op_name);
    for (auto it = range.first; it != range.second; ++it) {
      ORT_ENFORCE(std::get<0>(it->second) != name_with_ver_range);
    }

    // Parse start and end versions.
    std::stringstream ss(name_with_ver_range);
    std::string segment;
    std::vector<std::string> segments;
    while (std::getline(ss, segment, '_')) {
      segments.push_back(segment);
    }

    size_t len = segments.size();
    ORT_ENFORCE(len > 2);
    int start_ver = std::stoi(segments[len - 2]);
    int end_ver = segments[len - 1] == "INF" ? INT_MAX : std::stoi(segments[len - 1]);
    op_to_ver_range_map_.emplace(op_name, std::make_tuple(name_with_ver_range, start_ver, end_ver));
    Register<DerivedType>(name_with_ver_range);
  }

  GradientDef GetGradientForOp(const GradientGraphConfiguration& gradient_graph_config, Graph* graph, const Node* node,
                               int max_version, const std::unordered_set<std::string>& output_args_need_grad,
                               const std::unordered_set<std::string>& input_args_need_grad,
                               const logging::Logger& logger);

 private:
  GradientBuilderRegistry() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GradientBuilderRegistry);

  std::multimap<std::string, std::tuple<std::string, int, int>> op_to_ver_range_map_;
};

}  // namespace training
}  // namespace onnxruntime
