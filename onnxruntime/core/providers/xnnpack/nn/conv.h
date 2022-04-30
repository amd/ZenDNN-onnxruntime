// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {

class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info);

  Status Compute(OpKernelContext* /*context*/) const override;

  // use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  // due to other constraints of this kernel the value of group is either 1 or C, so we can infer that if it's not 1
  // it's a depthwise convolution
  // TODO technically C could be 1. Do we need to handle that or does it not matter in that case and
  //                                standard == depthwise if C == 1? Guessing that is the case for now. TBC
  bool IsDepthwise() const { return conv_attrs_.group != 1; }

  ConvAttributes conv_attrs_;
  TensorShapeVector kernel_shape_;
  int64_t M_;
  std::unique_ptr<Tensor> packed_w_;
  std::optional<std::pair<float, float>> clip_min_max_;

  XnnpackOperator op0_ = nullptr;
  // TensorShape output_shape_;
  // bool has_const_output_shape_;
  AllocatorPtr cpu_allocator_;

  // weight for depthwise. TBD if we need this. Can handling in PrePack though and use a unique_ptr
  // float* weight_ = nullptr;

  // The following vars are valid only when has_const_output_shape_ == false;
  struct Attrs {
    uint32_t input_padding_top_ = 0;
    uint32_t input_padding_right_ = 0;
    uint32_t input_padding_bottom_ = 0;
    uint32_t input_padding_left_ = 0;
    uint32_t subsampling_height_ = 0;
    uint32_t subsampling_width_ = 0;
    uint32_t dilation_height_ = 0;
    uint32_t dilation_width_ = 0;
    int padding_mode_ = 0;
  };
  std::optional<Attrs> attrs_;
};

}  // namespace xnnpack
}  // namespace onnxruntime
