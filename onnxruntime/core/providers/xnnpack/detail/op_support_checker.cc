// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_support_checker.h"

#include <unordered_map>

#include "core/common/common.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace xnnpack {

namespace {
bool IsPaddingTypeSupported(AutoPadType auto_pad) {
  return auto_pad == AutoPadType::NOTSET ||
         auto_pad == AutoPadType::VALID ||
         auto_pad == AutoPadType::SAME_UPPER;
}

// function to check if a node is supported.
// supported_nodes are previously selected nodes so that we can check for connected activation nodes that an L2 fusion
// could combine
using CheckerFn = std::function<bool(const Node& node, bool matched_kernel,
                                     const GraphViewer& graph,
                                     const std::unordered_set<const Node*>& supported_nodes)>;

// check if the details of Conv are supported.
//
// this is NOT a check that attributes etc. match the Conv spec. the kernel implementation can do that (assumably
// via some common code shared by all kernels for Conv). this is more consistent as we don't do checking of
// attributes etc. when matching a kernel for the CPU EP.
//
// we already validated type constraints via the kernel lookup in GetCapability so we know we're dealing with
// float input.
bool ConvChecker(const Node& node, bool matched_kernel, const GraphViewer& graph,
                 const std::unordered_set<const Node*>& /*supported_nodes*/) {
  // require kernel match so we know type constraints etc. have been checked
  if (!matched_kernel) {
    return false;
  }

  bool supported = false;

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // Conv has at least 2 inputs.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];
    const auto& weight_arg = *input_defs[1];

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }

    // weight must be constant and also rank 4
    const auto* weight = graph.GetConstantInitializer(weight_arg.Name(), true);
    if (weight == nullptr || weight->dims_size() != 4) {
      break;
    }

    ProtoHelperNodeContext nc(node);
    OpNodeProtoHelper info(&nc);

    // based on the PR the 'group' value needs to be 1 or C.
    // the second dim of weight is C/group, so if that == 1, group == C
    int64_t group = 0;
    info.GetAttrOrDefault<int64_t>("group", &group, 1);
    if (group != 1 && weight->dims(1) != 1) {
      break;
    }

    // if 'pads' is not specified we use 'auto_pad'
    if (graph_utils::GetNodeAttribute(node, "pads") == nullptr) {
      AutoPadType auto_pad = AutoPadType::NOTSET;

      std::string auto_pad_str;
      if (info.GetAttr<std::string>("auto_pad", &auto_pad_str).IsOK()) {
        // auto_pad was set
        //
        // The "auto_pad_str" string must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
        // tf2onnx converter doesn't use SAME_LOWER.
        // SAME_UPPER maps to TF SAME padding.
        // TODO: What does PT converter use? We need to support models from PT in mobile.
        // TODO: Can xnnpack support SAME_LOWER?

        auto_pad = StringToAutoPadType(auto_pad_str);
        if (!IsPaddingTypeSupported(auto_pad)) {
          break;
        }
      }
    }

    supported = true;
  } while (false);

  return supported;
}

bool ClipReluChecker(const Node& node, bool matched_kernel,
                     const GraphViewer& graph,
                     const std::unordered_set<const Node*>& supported_nodes) {
  assert(!matched_kernel);  // we don't have a Clip kernel - temporary sanity check

  bool supported = false;

  do {
    if (node.Domain() != kOnnxDomain) {
      break;
    }

    // input 0 must come from a Conv we support
    const Node::EdgeEnd* input0_edge = graph_utils::GetInputEdge(node, 0);
    if (!input0_edge) {
      break;
    }

    const Node& input0 = input0_edge->GetNode();
    if (input0.OpType() != "Conv" || supported_nodes.count(&input0) == 0) {
      break;
    }

    // if Clip check the min/max are constant.
    if (node.OpType() == "Clip") {
      const auto& input_args = node.InputDefs();
      const auto num_inputs = input_args.size();
      if (num_inputs >= 2) {
        // check 'min' is constant
        if (!graph.IsConstantInitializer(input_args[1]->Name(), true)) {
          break;
        }
      }

      if (num_inputs == 3) {
        // check 'max' is constant
        if (!graph.IsConstantInitializer(input_args[2]->Name(), true)) {
          break;
        }
      }
    }

    supported = true;
  } while (false);

  return supported;
}

}  // namespace

bool NodeSupportChecker::IsNodeSupported(const Node& node, bool matched_kernel) {
  static std::unordered_map<std::string, CheckerFn> checkers{
      {"Conv", ConvChecker},
      {"Clip", ClipReluChecker},  // testing fusion of Conv+Activation with min/max
      {"Relu", ClipReluChecker},  // testing fusion of Conv+Activation
  };

  const auto entry = checkers.find(node.OpType());
  bool supported = false;
  if (entry != checkers.cend()) {
    supported = entry->second(node, matched_kernel, graph_, supported_nodes_);
  }

  return supported;
}
}  // namespace xnnpack
}  // namespace onnxruntime
