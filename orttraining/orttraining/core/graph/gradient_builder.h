// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "gradient_builder_base.h"

namespace onnxruntime {
namespace training {
// TODO: maybe group the gradient builders and split them into different files.
#define DECLARE_GRADIENT_BUILDER(name, start_ver, end_ver)            \
  class name##_##start_ver##_##end_ver : public GradientBuilderBase { \
    using GradientBuilderBase::GradientBuilderBase;                   \
    std::vector<NodeDef> GetGradientDefsImpl() const override;        \
  };

DECLARE_GRADIENT_BUILDER(GetCastGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSinGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetLogGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetTanhGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSqrtGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetErfGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetMatMulGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSplitGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetReluGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetAddSubGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetMulGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetDivGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetNegGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetReduceMeanGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetReduceSumGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetReduceLogSumExpGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetReduceL2Gradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetPowGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetConcatGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetConcatTrainingGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetReshapeGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetTransposeGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetPoolGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetAveragePoolGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetMaxPoolGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetGatherGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetConvGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetUnsqueezeGradient, 1, 12)
DECLARE_GRADIENT_BUILDER(GetUnsqueezeGradient, 13, INF)
DECLARE_GRADIENT_BUILDER(GetSqueezeGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSigmoidGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSoftmaxGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetLogSoftmaxGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSoftmaxCrossEntropyGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSparseSoftmaxCrossEntropyGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSoftmaxCrossEntropyLossGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSoftmaxCrossEntropyLossInternalGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetGlobalAveragePoolGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetGemmGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetDropoutGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetGatherNDGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetGatherElementsGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetGeluGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetBiasGeluGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetFastGeluGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetLayerNormalizationGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSimplifiedLayerNormalizationGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetBatchNormalizationGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetMegatronFGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetMegatronGGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSliceGradient, 10, INF)
DECLARE_GRADIENT_BUILDER(GetWhereGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetSendGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetRecvGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetExpandGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetExpGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetFlattenGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetTopKGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetClipGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetAbsGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetMinMaxGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetTileGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetATenOpGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetPadGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetIdentityGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetPythonOpGradient, 1, INF)
DECLARE_GRADIENT_BUILDER(GetScatterNDGradient, 1, INF)

}  // namespace training
}  // namespace onnxruntime
