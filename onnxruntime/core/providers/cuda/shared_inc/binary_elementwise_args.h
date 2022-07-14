// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

enum class PerChannelType : int32_t {
  None = (int32_t)0,
  LhsNeedCompute = (int32_t)1,
  RhsNeedCompute = (int32_t)2,
};

struct BinaryElementwiseArgs {
  BroadcastIndexType lhs_index_type;
  BroadcastIndexType rhs_index_type;
  size_t output_size;

  // Required if any index type is NeedCompute.
  size_t rank;
  TArray<int64_t> lhs_strides;
  TArray<int64_t> rhs_strides;
  TArray<fast_divmod> output_fdms;

  // Optimization for case Op([N,C,H],[C,1]).
  PerChannelType per_channel_type{PerChannelType::None};
  bool is_multi_batch;
  int channel;
  int height;

  static BinaryElementwiseArgs NoBroadcastArgs(size_t output_size) {
    BinaryElementwiseArgs args;
    args.lhs_index_type = args.rhs_index_type = BroadcastIndexType::NoBroadcast;
    args.output_size = output_size;
    return args;
  }
};

}  // namespace cuda
}  // namespace onnxruntime
