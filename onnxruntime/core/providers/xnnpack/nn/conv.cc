// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"
#include "core/graph/constants.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/framework/tensorprotoutils.h"
namespace onnxruntime {
namespace xnnpack {

namespace {
Status CreateXnnpackKernel(const ConvAttributes& conv_attrs,
                           int64_t C, int64_t M,
                           const TensorShapeVector& kernel_shape,
                           const std::optional<std::pair<float, float>>& clip_min_max,
                           bool depthwise,
                           const float* W_data, const float* B_data,
                           struct xnn_operator*& p) {
  // X input is NHWC
  // int64_t H = X_shape[1];
  // int64_t W = X_shape[2];
  // int64_t C = X_shape[3];       // input_channels
  // int64_t M = kernel_shape[0];  // output_channels

  // copied from logic in xnnpack_transformer.cc
  // int64_t output_channels = kernel_shape[0]; -- this is misleading as 'kernel_shape' was really weight shape
  uint32_t kernel_height = gsl::narrow<uint32_t>(kernel_shape[0]);
  uint32_t kernel_width = gsl::narrow<uint32_t>(kernel_shape[1]);

  uint32_t input_padding_top = gsl::narrow<uint32_t>(conv_attrs.pads[0]);
  uint32_t input_padding_left = gsl::narrow<uint32_t>(conv_attrs.pads[1]);
  uint32_t input_padding_bottom = gsl::narrow<uint32_t>(conv_attrs.pads[2]);
  uint32_t input_padding_right = gsl::narrow<uint32_t>(conv_attrs.pads[3]);

  uint32_t subsampling_height = gsl::narrow<uint32_t>(conv_attrs.strides[0]);
  uint32_t subsampling_width = gsl::narrow<uint32_t>(conv_attrs.strides[1]);
  uint32_t dilation_height = gsl::narrow<uint32_t>(conv_attrs.dilations[0]);
  uint32_t dilation_width = gsl::narrow<uint32_t>(conv_attrs.dilations[1]);

  float output_min = clip_min_max ? clip_min_max->first : std::numeric_limits<float>::min();
  float output_max = clip_min_max ? clip_min_max->second : std::numeric_limits<float>::max();

  uint32_t flags = 0;
  if (conv_attrs.auto_pad == AutoPadType::SAME_UPPER) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  xnn_status status = xnn_status::xnn_status_uninitialized;
  p = nullptr;

  /*
  From PR

  Standard:
      size_t group_input_channels = input_channels / groups;
      size_t group_output_channels = output_channels / groups;
      status = xnn_create_convolution2d_nhwc_f32(
          input_padding_top_, input_padding_right_, input_padding_bottom_, input_padding_left_,
          gsl::narrow<uint32_t>(kernel_height), gsl::narrow<uint32_t>(kernel_width), subsampling_height_,
          subsampling_width_, dilation_height_, dilation_width_, gsl::narrow<uint32_t>(groups),
          gsl::narrow<uint32_t>(group_input_channels), gsl::narrow<uint32_t>(group_output_channels),
          gsl::narrow<uint32_t>(input_channels), gsl::narrow<uint32_t>(output_channels), weight->Data<float>(),
          B->Data<float>(), output_min, output_max, flags, &p);
      ORT_ENFORCE(status == xnn_status_success);
      op0_.reset(p);

  Depthwise:

      int64_t depth_multiplier = kernel_shape[3] / input_channels; // 'kernel_shape' is actual the weight shape
      xnn_status status = xnn_create_convolution2d_nhwc_f32(
          input_padding_top_, input_padding_right_, input_padding_bottom_, input_padding_left_,
          gsl::narrow<uint32_t>(kernel_height), gsl::narrow<uint32_t>(kernel_width), subsampling_height_,
          subsampling_width_, dilation_height_, dilation_width_, gsl::narrow<uint32_t>(input_channels) * groups *,
          1 * group_input_channels *, depth_multiplier * group_output_channels *,
          input_channels *input_channel_stride*, kernel_shape[3] *output_channel_stride*,
          weight_, B->Data<float>(),
          output_min, output_max, flags, &p);
      ORT_ENFORCE(status == xnn_status_success);
      op0_.reset(p);

  */

  if (depthwise) {
    ORT_ENFORCE(conv_attrs.group == C);
    // C == group and M % group == 0 so this should result in a whole number
    uint32_t group_output_channels = gsl::narrow<uint32_t>(M / conv_attrs.group);

    status = xnn_create_convolution2d_nhwc_f32(
        input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
        kernel_height, kernel_width,
        subsampling_height, subsampling_width,
        dilation_height, dilation_width,
        // groups, group_input_channels, group_output_channels
        gsl::narrow<uint32_t>(conv_attrs.group), 1, group_output_channels,
        // TODO: Not sure it's correct for depthwise and standard conv to have the same values here
        // Does conv_attrs.stride potentially need to be taken into account?
        // input channel stride, output channel stride
        C, M,
        W_data, B_data,
        output_min, output_max, flags, &p);

  } else {
    // original code had the below, but as group == 1 if this is not depthwise that seems unnecessary if you're always
    // dividing by 1.
    //
    // uint32_t group_input_channels = gsl::narrow<uint32_t>(C / conv_attrs.group);
    // uint32_t group_output_channels = gsl::narrow<uint32_t>(M / conv_attrs.group);
    ORT_ENFORCE(conv_attrs.group == 1);

    // TODO: What is the cost of this call?
    // If B is not a constant initializer we have to do this on every call to Compute.
    // Is that viable or do we need to constrain usage of xnnpack to nodes where B is constant or not specified?
    status = xnn_create_convolution2d_nhwc_f32(
        input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
        kernel_height, kernel_width,
        subsampling_height, subsampling_width,
        dilation_height, dilation_width,
        gsl::narrow<uint32_t>(conv_attrs.group), C, M,  // groups, group_input_channels, group_output_channels
        C, M,                                           // input channel stride, output channel stride
        W_data, B_data,
        output_min, output_max, flags, &p);
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create xnnpack kernel. xnn_create_convolution2d_nhwc_f32 returned ", status);
  }

  return Status::OK();
}
}  // namespace

Conv::Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_{info} {
  // get values from any fusion with an activation
  if (info.GetAttr<std::string>("activation", &conv_attrs_.activation).IsOK()) {
    std::vector<float> activation_params;

    // min/max could be from Clip or Relu
    if (info.GetAttrs<float>("activation_params", activation_params).IsOK()) {
      if (activation_params.size() == 2) {
        clip_min_max_ = {activation_params[0], activation_params[1]};
      }
    }
  }

  const auto& node{Node()};

  // as the weight input is a constant initializer we can calculate all the sizes here instead of in Compute
  const Tensor* W = nullptr;
  ORT_ENFORCE(info.TryGetConstantInput(1, &W),
              "Weight input was not constant initializer. XNNPACK EP should not have asked for the node."
              "Node name:",
              node.Name());

  // 'M' is first dim of weight. Prepacking will alter the layout of W so save to member for use in Compute
  M_ = W->Shape()[0];

  // this happens before PrePack, so the W input is still in the ONNX spec format
  ORT_THROW_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape_));

  // TODO: Add method to ConvAttributes to do this initialization once kernel shape is known.
  // Or do it in ComputeKernelShape
  if (conv_attrs_.pads.empty()) {
    conv_attrs_.pads.resize(kernel_shape_.size() * 2, 0);
  }

  if (conv_attrs_.dilations.empty()) {
    conv_attrs_.dilations.resize(kernel_shape_.size(), 1);
  }

  if (conv_attrs_.strides.empty()) {
    conv_attrs_.strides.resize(kernel_shape_.size(), 1);
  }

  // we only take nodes with no bias, or a constant bias.
  const Tensor* B = nullptr;
  const auto& input_defs = node.InputDefs();
  bool has_bias = input_defs.size() == 3 && input_defs[2]->Exists();

  ORT_ENFORCE(has_bias == false || info.TryGetConstantInput(2, &B),
              "Invalid Node with non-constant Bias input. XNNPACK EP should not have asked for the node. Node name:",
              node.Name());

  // we know we have the C, H, W values for this kernel to be used due to checks in ConvChecker.
  const NodeArg& X = *input_defs[0];
  TensorShape X_shape = utils::GetTensorShapeFromTensorShapeProto(*X.Shape());
  int64_t C = X_shape[3];  // input is NHWC

  struct xnn_operator* p = nullptr;
  ORT_THROW_IF_ERROR(CreateXnnpackKernel(conv_attrs_, C, M_, kernel_shape_, clip_min_max_, IsDepthwise(),
                                         W->Data<float>(), B ? B->Data<float>() : nullptr, p));

  op0_.reset(p);
}

// use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
Status Conv::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                     /*out*/ bool& is_packed,
                     /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;

  // only layout of weight input is adjusted via PrePack
  if (input_idx == 1) {
    auto orig_shape = tensor.Shape();
    std::cout << "Got PrePack with W shape of " << orig_shape << "\n";

    std::vector<int64_t> perm;
    perm.reserve(4);

    // This is transpose of weights in PR
    //  std::vector<int64_t> weight_perm =
    //        is_depthwise ? std::vector<int64_t>{1, 2, 3, 0}  // {C/group, kH, kW, M}. group == C so -> {1, kH, kW, M}
    //                     : std::vector<int64_t>{0, 2, 3, 1}; // [M, kH, kW, C/group}. group == 1 so -> {M, kH, kW, C}

    // copied from PR
    if (IsDepthwise()) {
      perm = {1, 2, 3, 0};
    } else {
      perm = {0, 2, 3, 1};
    }

    // TODO: transpose weight using perm and the common transpose helper

    // NOTE: For this demo create a Tensor for the packed weight so there's a shape attached.
    //       Could alternatively create a raw buffer with IAllocator::MakeUniquePtr<void> if the overhead of a Tensor
    //       is not needed.

    // arbitrary example moving first dim to the end - which coincidentally is the depthwise transpose required
    // in the real implementation the transpose of the data would also be done.
    std::vector<int64_t> new_shape;
    auto rank = orig_shape.NumDimensions();
    new_shape.reserve(rank);

    if (IsDepthwise()) {
      new_shape.push_back(orig_shape[1]);
      new_shape.push_back(orig_shape[2]);
      new_shape.push_back(orig_shape[3]);
      new_shape.push_back(orig_shape[0]);
    } else {
      new_shape.push_back(orig_shape[0]);
      new_shape.push_back(orig_shape[2]);
      new_shape.push_back(orig_shape[3]);
      new_shape.push_back(orig_shape[1]);
    }

    packed_w_ = Tensor::Create(tensor.DataType(), TensorShape(new_shape), alloc);

    // set to arbitrary value for now
    memset(packed_w_->MutableDataRaw(), 7, packed_w_->SizeInBytes());

    is_packed = true;
  }

  return Status::OK();
}

Status Conv::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor& X = *context->Input<Tensor>(0);  // this is in NHWC format
  // const Tensor* B = num_inputs >= 3 ? context->Input<Tensor>(2) : nullptr;

  if (num_inputs == 3 && !op0_) {
    ORT_NOT_IMPLEMENTED(
        "TODO: Could/should we support creating an xnnpack kernel on every call to Compute? "
        "The bias input is required at kernel creation time.");
  }

  const auto& X_shape = X.Shape();
  std::cout << "Xnnpack Conv::Compute called with input shape of " << X_shape << "\n";

  const int64_t N = X_shape[0];  // input is NHWC
  const int64_t H = X_shape[1];
  const int64_t W = X_shape[2];
  // const int64_t C = X_shape[3];
  const int64_t M = M_;

  // We don't need to call ValidateInputShape as we checked validity in ConvChecker.
  // We also can't use it as-is as the weight tensor was pre-packed and the layout was changed there.
  // ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&X, &W));

  TensorShapeVector Y_dims({N, M});
  TensorShape input_shape = X.Shape().Slice(2);

  // TODO: Is the implementation/result of this any different to what XnnPackConvShapeInferKernelImpl or
  // XnnPackDepthwiseConvolution2dShapeInferImpl would produce?
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape_,
                                                   conv_attrs_.strides, conv_attrs_.dilations, conv_attrs_.pads,
                                                   Y_dims));

  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      op0_.get(), N, H, W, X.Data<float>(), Y->MutableData<float>(),
      nullptr /*threadpool*/);  // TBD: how to handle threading

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_convolution2d_nhwc_f32 returned ", status);
  }

  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Conv, kMSInternalNHWCDomain, 1, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  Conv);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Conv, kOnnxDomain, 1, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  utils::InvalidNchwKernel);

ONNX_OPERATOR_KERNEL_EX(Conv, kMSInternalNHWCDomain, 11, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        Conv);
ONNX_OPERATOR_KERNEL_EX(Conv, kOnnxDomain, 11, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        utils::InvalidNchwKernel);

}  // namespace xnnpack
}  // namespace onnxruntime
