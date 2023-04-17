// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/rule_based_graph_transformer.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

common::Status GraphTransformerManager::SetSteps(unsigned steps) {
  steps_ = steps;
  return Status::OK();
}

common::Status GraphTransformerManager::GetSteps(unsigned& steps) const {
  steps = steps_;
  return Status::OK();
}

common::Status GraphTransformerManager::ApplyTransformers(Graph& graph, TransformerLevel level,
                                                          const logging::Logger& logger) const {
  const auto& transformers = level_to_transformer_map_.find(level);
  if (transformers == level_to_transformer_map_.end()) {
    return Status::OK();
  }

  for (unsigned step = 0; step < steps_; ++step) {
    bool graph_changed = false;
    for (const auto& transformer : transformers->second) {
      if (step > 0 && transformer->ShouldOnlyApplyOnce())
        continue;

      bool modified = false;
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, logger));
      graph_changed = graph_changed || modified;
    }
    if (!graph_changed) {
      break;
    }
  }

  return Status::OK();
}

common::Status GraphTransformerManager::Register(std::unique_ptr<GraphTransformer> transformer,
                                                 TransformerLevel level) {
  const auto& name = transformer->Name();
  auto& transformers_for_level = level_to_transformer_map_[level];

  // allow the transformer to be registered multiple times but only in different levels
  if (std::find_if(transformers_for_level.begin(), transformers_for_level.end(),
                   [&name](const auto& entry) { return entry->Name() == name; }) != transformers_for_level.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Transformer is already registered: " + name);
  }

  transformers_for_level.push_back(std::move(transformer));
  return Status::OK();
}
}  // namespace onnxruntime
