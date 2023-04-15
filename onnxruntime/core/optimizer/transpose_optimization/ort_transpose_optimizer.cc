// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/transpose_optimization/ort_transpose_optimizer.h"
#include "core/graph/constants.h"

using namespace onnx_transpose_optimization;

namespace onnxruntime {

// Special case For CPU EP where we can potentially replace with NhwcMaxPool.
// Only int8 and uint8 dtypes are supported by NhwcMaxPool.
static bool HandleMaxPool(HandlerArgs& args) {
  if (args.node.GetExecutionProviderType() != "CPUExecutionProvider") {
    return false;
  }

  auto outputs = args.node.Outputs();
  if (outputs.size() == 2 && outputs[1] != "") {
    // Can't optimize if optional "indices" output is provided
    return false;
  }

  auto info = args.ctx.graph.GetValueInfo(outputs[0]);
  api::DataType dtype = info->DType();
  if (dtype != api::DataType::UINT8 && dtype != api::DataType::INT8) {
    return false;
  }

  size_t rank = args.perm.size();
  if (args.perm != ChannelLastToFirstPerm(rank)) {
    return false;
  }

  auto new_node = SwapNodeOpTypeDomainAndSinceVersion(args.ctx.graph, args.node, "NhwcMaxPool", "com.microsoft", 1);
  new_node->ClearAttribute("storage_order");  // Only relevant for indices output. Prohibited for NhwcMaxPool.
  TransposeFirstInput(args.ctx, *new_node, args.perm_inv);
  TransposeOutputs(args.ctx, *new_node, args.perm);
  return true;
}

static bool HandleQLinearConcat(HandlerArgs& args) {
  return HandleSimpleNodeWithAxis(args);
}

std::vector<size_t> QLinearConcatInputs(OptimizerCtx& ctx, api::NodeRef& node) {
  (void)ctx;
  std::vector<size_t> indices;
  size_t num_inputs = node.Inputs().size();
  for (size_t i = 2; i < num_inputs; i += 3) {
    indices.push_back(i);
  }
  return indices;
}

constexpr HandlerInfo q_linear_concat_handler = {&QLinearConcatInputs, &HandleQLinearConcat};

static bool HandleQLinearBinaryOp(HandlerArgs& args) {
  return HandleSimpleNodeBroadcast(args);
}

std::vector<size_t> QLinearBinaryOpInputs(OptimizerCtx&, api::NodeRef&) {
  // Inputs are: [A, A_scale, A_zero_point, B, B_scale, B_zero_point, C_scale, C_zero_point],
  // we want [A, B].
  return {0, 3};
}

constexpr HandlerInfo q_linear_binary_op_handler = {&QLinearBinaryOpInputs, &HandleQLinearBinaryOp};

static bool HandleQLinearPoolOp(HandlerArgs& args) {
  // Swap between channel first/last variants. Only works for applicable values of perm.
  int64_t channels_last = args.node.GetAttributeIntDefault("channels_last", 0);
  size_t rank = args.perm.size();
  if (rank < 2) return false;
  auto p = ChannelLastToFirstPerm(rank);
  if ((!channels_last && args.perm == p) || (channels_last && args.perm_inv == p)) {
    args.node.SetAttributeInt("channels_last", 1 - channels_last);
    TransposeFirstInput(args.ctx, args.node, args.perm_inv);
    TransposeOutputs(args.ctx, args.node, args.perm);
    return true;
  }
  return false;
}

constexpr HandlerInfo q_linear_pool_op_handler = {&FirstInput, &HandleQLinearPoolOp};

// ops using base handler implementations
constexpr HandlerInfo max_pool_op_handler = {&FirstInput, &HandleMaxPool};
constexpr HandlerInfo node_1_inp_handler = {&FirstInput, &HandleSimpleNode};
constexpr HandlerInfo reduce_op_handler = {&FirstInput, &HandleReduceOps};

// ORT contrib ops and special cased ONNX ops
const HandlerMap& OrtExtendedHandlers() {
  static const HandlerMap extended_handler_map{
      {"com.microsoft.QLinearReduceMean", reduce_op_handler},
      {"com.microsoft.QLinearSigmoid", node_1_inp_handler},
      {"com.microsoft.QLinearLeakyRelu", node_1_inp_handler},
      {"com.microsoft.QLinearConcat", q_linear_concat_handler},
      {"com.microsoft.QLinearAdd", q_linear_binary_op_handler},
      {"com.microsoft.QLinearMul", q_linear_binary_op_handler},
      {"com.microsoft.QLinearAveragePool", q_linear_pool_op_handler},
      {"com.microsoft.QLinearGlobalAveragePool", q_linear_pool_op_handler},
      {"MaxPool", max_pool_op_handler},
      // TODO: Add Resize. Figure out how/where to do that
  };

  return extended_handler_map;
}

CostCheckResult OrtEPCostCheck(const api::GraphRef& graph, const api::NodeRef& node,
                               const std::vector<int64_t>& /*perm*/,
                               const std::unordered_set<std::string>& /*outputs_leading_to_transpose*/) {
  // special case some kernels based on the ORT implementation details
  if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
    if (node.IsOp("MaxPool")) {
      // MaxPool has higher perf in the NHWC variant when supported. HandleMaxPool does the support checks.
      return CostCheckResult::kPushTranspose;
    }

    if (node.IsOp("Resize")) {
      // Resize is included because it has higher perf in the NHWC variant when
      // the input X is 4D int8 tensor and the mode is linear
      auto X_value_info = graph.GetValueInfo(node.Inputs()[0]);
      auto X_shape = X_value_info->Shape();
      auto X_dtype = X_value_info->DType();
      auto mode = node.GetAttributeString("mode");
      if (X_shape && X_shape->size() == 4 &&
          (X_dtype == api::DataType::UINT8 || X_dtype == api::DataType::INT8) &&
          mode && *mode == "linear") {
        return CostCheckResult::kPushTranspose;
      }
    }
  }

  return CostCheckResult::kFallThrough;
}

}  // namespace onnxruntime
