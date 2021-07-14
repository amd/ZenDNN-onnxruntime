// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/gradient_builder.h"
#include "orttraining/core/graph/gradient_config.h"

namespace onnxruntime {
namespace training {

GradientDef GradientBuilderRegistry::GetGradientForOp(const GradientGraphConfiguration& gradient_graph_config,
                                                      Graph* graph, const Node* node, int max_version,
                                                      const std::unordered_set<std::string>& output_args_need_grad,
                                                      const std::unordered_set<std::string>& input_args_need_grad,
                                                      const logging::Logger& logger) {
  auto range = op_to_ver_range_map_.equal_range(node->OpType());
  std::string name_with_ver_range = "";
  for (auto it = range.first; it != range.second; ++it) {
    const auto& tuple = it->second;
    if (max_version >= std::get<1>(tuple) && max_version <= std::get<2>(tuple)) {
      name_with_ver_range = std::get<0>(tuple);
      break;
    }
  }

  auto gradient_builder = MakeUnique(name_with_ver_range, gradient_graph_config, graph, node, output_args_need_grad,
                                     input_args_need_grad, logger);

  ORT_ENFORCE(gradient_builder != nullptr, "The gradient builder has not been registered:", node->OpType(),
              " for node ", node->Name(), " in OpSet ", std::to_string(max_version));

  return gradient_builder->GetGradientDefs();
}

#define REGISTER_GRADIENT_BUILDER(op, start_ver, end_ver, gradientbuilder)                                   \
  GradientBuilderRegistry::GetInstance().RegisterGradientBuilder<gradientbuilder##_##start_ver##_##end_ver>( \
      op, std::string(op) + "_" + #start_ver + "_" + #end_ver);

#define NO_GRADIENT(op, start_ver, end_ver) REGISTER_GRADIENT_BUILDER(op, start_ver, end_ver, EmptyGradientBuilder)

// There are some operators which are not really computation operators and one shouldn't attempt to
// request one for such operators.
#define SHOULD_NOT_DO_GRADIENT(op, start_ver, end_ver) \
  REGISTER_GRADIENT_BUILDER(op, start_ver, end_ver, UnSupportedGradientBuilder)

void GradientBuilderRegistry::RegisterGradientBuilders() {
  // Register gradient builders here.
  REGISTER_GRADIENT_BUILDER("Cast", 1, INF, GetCastGradient);
  REGISTER_GRADIENT_BUILDER("Sin", 1, INF, GetSinGradient);
  REGISTER_GRADIENT_BUILDER("Log", 1, INF, GetLogGradient);
  REGISTER_GRADIENT_BUILDER("Tanh", 1, INF, GetTanhGradient);
  REGISTER_GRADIENT_BUILDER("Sqrt", 1, INF, GetSqrtGradient);
  REGISTER_GRADIENT_BUILDER("Erf", 1, INF, GetErfGradient);
  REGISTER_GRADIENT_BUILDER("MatMul", 1, INF, GetMatMulGradient);
  REGISTER_GRADIENT_BUILDER("Split", 1, INF, GetSplitGradient);
  REGISTER_GRADIENT_BUILDER("Relu", 1, INF, GetReluGradient);
  REGISTER_GRADIENT_BUILDER("Pow", 1, INF, GetPowGradient);
  REGISTER_GRADIENT_BUILDER("ReduceMean", 1, INF, GetReduceMeanGradient);
  REGISTER_GRADIENT_BUILDER("ReduceSum", 1, INF, GetReduceSumGradient);
  REGISTER_GRADIENT_BUILDER("ReduceLogSumExp", 1, INF, GetReduceLogSumExpGradient);
  REGISTER_GRADIENT_BUILDER("ReduceL2", 1, INF, GetReduceL2Gradient);
  REGISTER_GRADIENT_BUILDER("Add", 1, INF, GetAddSubGradient);
  REGISTER_GRADIENT_BUILDER("Sub", 1, INF, GetAddSubGradient);
  REGISTER_GRADIENT_BUILDER("Mul", 1, INF, GetMulGradient);
  REGISTER_GRADIENT_BUILDER("Div", 1, INF, GetDivGradient);
  REGISTER_GRADIENT_BUILDER("Neg", 1, INF, GetNegGradient);
  REGISTER_GRADIENT_BUILDER("Concat", 1, INF, GetConcatGradient);
  REGISTER_GRADIENT_BUILDER("ConcatTraining", 1, INF, GetConcatTrainingGradient);
  REGISTER_GRADIENT_BUILDER("Reshape", 1, INF, GetReshapeGradient);
  REGISTER_GRADIENT_BUILDER("Transpose", 1, INF, GetTransposeGradient);
  REGISTER_GRADIENT_BUILDER("Gemm", 1, INF, GetGemmGradient);
  REGISTER_GRADIENT_BUILDER("MaxPool", 1, INF, GetMaxPoolGradient);
  REGISTER_GRADIENT_BUILDER("Gather", 1, INF, GetGatherGradient);
  REGISTER_GRADIENT_BUILDER("Conv", 1, INF, GetConvGradient);
  REGISTER_GRADIENT_BUILDER("Squeeze", 1, INF, GetSqueezeGradient);
  REGISTER_GRADIENT_BUILDER("Unsqueeze", 1, 12, GetUnsqueezeGradient);
  REGISTER_GRADIENT_BUILDER("Unsqueeze", 13, INF, GetUnsqueezeGradient);
  REGISTER_GRADIENT_BUILDER("Sigmoid", 1, INF, GetSigmoidGradient);
  REGISTER_GRADIENT_BUILDER("Softmax", 1, INF, GetSoftmaxGradient);
  REGISTER_GRADIENT_BUILDER("LogSoftmax", 1, INF, GetLogSoftmaxGradient);
  REGISTER_GRADIENT_BUILDER("SoftmaxCrossEntropy", 1, INF, GetSoftmaxCrossEntropyGradient);
  REGISTER_GRADIENT_BUILDER("SparseSoftmaxCrossEntropy", 1, INF, GetSparseSoftmaxCrossEntropyGradient);
  REGISTER_GRADIENT_BUILDER("SoftmaxCrossEntropyLoss", 1, INF, GetSoftmaxCrossEntropyLossGradient);
  REGISTER_GRADIENT_BUILDER("SoftmaxCrossEntropyLossInternal", 1, INF, GetSoftmaxCrossEntropyLossInternalGradient);
  REGISTER_GRADIENT_BUILDER("GlobalAveragePool", 1, INF, GetGlobalAveragePoolGradient);
  REGISTER_GRADIENT_BUILDER("AveragePool", 1, INF, GetAveragePoolGradient);
  REGISTER_GRADIENT_BUILDER("Dropout", 1, INF, GetDropoutGradient)
  REGISTER_GRADIENT_BUILDER("GatherND", 1, INF, GetGatherNDGradient)
  REGISTER_GRADIENT_BUILDER("GatherElements", 1, INF, GetGatherElementsGradient)
  REGISTER_GRADIENT_BUILDER("Gelu", 1, INF, GetGeluGradient)
  REGISTER_GRADIENT_BUILDER("BiasGelu", 1, INF, GetBiasGeluGradient);
  REGISTER_GRADIENT_BUILDER("FastGelu", 1, INF, GetFastGeluGradient);
  REGISTER_GRADIENT_BUILDER("LayerNormalization", 1, INF, GetLayerNormalizationGradient);
  REGISTER_GRADIENT_BUILDER("SimplifiedLayerNormalization", 1, INF, GetSimplifiedLayerNormalizationGradient);
  REGISTER_GRADIENT_BUILDER("BatchNormInternal", 1, INF, GetBatchNormalizationGradient);
  REGISTER_GRADIENT_BUILDER("MegatronF", 1, INF, GetMegatronFGradient);
  REGISTER_GRADIENT_BUILDER("MegatronG", 1, INF, GetMegatronGGradient);
  REGISTER_GRADIENT_BUILDER("Slice", 10, INF, GetSliceGradient);
  REGISTER_GRADIENT_BUILDER("Where", 1, INF, GetWhereGradient);
  REGISTER_GRADIENT_BUILDER("Send", 1, INF, GetSendGradient);
  REGISTER_GRADIENT_BUILDER("Recv", 1, INF, GetRecvGradient);
  REGISTER_GRADIENT_BUILDER("Expand", 1, INF, GetExpandGradient);
  REGISTER_GRADIENT_BUILDER("Exp", 1, INF, GetExpGradient);
  REGISTER_GRADIENT_BUILDER("Flatten", 1, INF, GetFlattenGradient);
  REGISTER_GRADIENT_BUILDER("TopK", 1, INF, GetTopKGradient);
  REGISTER_GRADIENT_BUILDER("Clip", 1, INF, GetClipGradient);
  REGISTER_GRADIENT_BUILDER("Abs", 1, INF, GetAbsGradient);
  REGISTER_GRADIENT_BUILDER("Min", 1, INF, GetMinMaxGradient);
  REGISTER_GRADIENT_BUILDER("Max", 1, INF, GetMinMaxGradient);
  REGISTER_GRADIENT_BUILDER("Tile", 1, INF, GetTileGradient);
  REGISTER_GRADIENT_BUILDER("ATenOp", 1, INF, GetATenOpGradient);
  REGISTER_GRADIENT_BUILDER("Pad", 1, INF, GetPadGradient);
  REGISTER_GRADIENT_BUILDER("Identity", 1, INF, GetIdentityGradient);
  REGISTER_GRADIENT_BUILDER("PythonOp", 1, INF, GetPythonOpGradient);
  REGISTER_GRADIENT_BUILDER("ScatterND", 1, INF, GetScatterNDGradient);
};

}  // namespace training
}  // namespace onnxruntime
