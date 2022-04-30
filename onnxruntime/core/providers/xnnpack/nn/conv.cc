// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"
#include "core/graph/constants.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"

namespace onnxruntime {
namespace xnnpack {

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

  // as the weight input is a constant initializer we can calculate all the sizes here instead of in Compute
  const Tensor* W = nullptr;
  ORT_ENFORCE(info.TryGetConstantInput(1, &W),
              "Weight input was not constant initializer. XNNPACK EP should not have asked for the node.");

  // 'M' is first dim of weight. Prepacking will alter the layout of W so save to member for use in Compute
  M_ = W->Shape()[0];

  // this happens before PrePack, so the W input is still in the ONNX spec format
  ORT_THROW_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape_));

  if (conv_attrs_.pads.empty()) {
    conv_attrs_.pads.resize(kernel_shape_.size() * 2, 0);
  }

  if (conv_attrs_.dilations.empty()) {
    conv_attrs_.dilations.resize(kernel_shape_.size(), 1);
  }

  if (conv_attrs_.strides.empty()) {
    conv_attrs_.strides.resize(kernel_shape_.size(), 1);
  }

  if (IsDepthwise()) {
  } else {
    uint32_t flags = 0;
    if (padding_mode == 1) {
      flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
    }

    size_t group_input_channels = input_channels / groups;
    size_t group_output_channels = output_channels / groups;
    xnn_status status;
    struct xnn_operator* p;
    status = xnn_create_convolution2d_nhwc_f32(
        input_padding_top_, input_padding_right_, input_padding_bottom_, input_padding_left_,
        gsl::narrow<uint32_t>(kernel_height), gsl::narrow<uint32_t>(kernel_width), subsampling_height_,
        subsampling_width_, dilation_height_, dilation_width_, gsl::narrow<uint32_t>(groups),
        gsl::narrow<uint32_t>(group_input_channels), gsl::narrow<uint32_t>(group_output_channels),
        gsl::narrow<uint32_t>(input_channels), gsl::narrow<uint32_t>(output_channels), weight->Data<float>(),
        B->Data<float>(), output_min, output_max, flags, &p);
    ORT_ENFORCE(status == xnn_status_success);
    op0_.reset(p);
  }

  /*
  Standard:
    uint32_t flags = 0;
  if (padding_mode == 1) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  size_t group_input_channels = input_channels / groups;
  size_t group_output_channels = output_channels / groups;
  xnn_status status;
  struct xnn_operator* p;
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

    uint32_t flags = 0;
  if (padding_mode == 1) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  int64_t depth_multiplier = kernel_shape[3] / input_channels;
  struct xnn_operator* p;
  xnn_status status = xnn_create_convolution2d_nhwc_f32(
      input_padding_top_, input_padding_right_, input_padding_bottom_, input_padding_left_,
      gsl::narrow<uint32_t>(kernel_height), gsl::narrow<uint32_t>(kernel_width), subsampling_height_,
      subsampling_width_, dilation_height_, dilation_width_, gsl::narrow<uint32_t>(input_channels) * groups *,
      1 * group_input_channels *, depth_multiplier * group_output_channels *,
      input_channels * input_channel_stride *, kernel_shape[3] * output_channel_stride *, weight_, B->Data<float>(),
      output_min, output_max, flags, &p);
  ORT_ENFORCE(status == xnn_status_success);
  op0_.reset(p);

  */
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
  const Tensor& W = *packed_w_;
  const Tensor* B = num_inputs >= 3 ? context->Input<Tensor>(2) : nullptr;

  const auto& X_shape = X.Shape();
  std::cout << "Xnnpack Conv::Compute called with input shape of " << X_shape << "\n";

  const int64_t N = X.Shape()[0];
  const int64_t C = X.Shape()[3];  // input is NHWC
  const int64_t M = M_;

  // We don't need this as we checked validity in ConvChecker. We also can't use it as-is as the weight tensor was
  // pre-packed so the layout won't necessarily match the ONNX Conv spec.
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

  xnn_status status = xnn_status_uninitialized;
  if (IsDepthwise()) {
  } else {
    xnn_status status = xnn_setup_convolution2d_nhwc_f32(
        op0_.get(), N * batch size*, input_shape[1] * input height*, input_shape[2] * input width*,
        X->Data<float>() * input*, Y->MutableData<float>() * output*, nullptr * threadpool*);
  }

  if (status != xnn_status_success) {
  }
  status = xnn_run_operator(op0_.get(), nullptr);
  ORT_ENFORCE(status == xnn_status_success);
  return Status::OK();

  ORT_NOT_IMPLEMENTED("TODO: add NHWC implementation here.");

  /*
  *
  *   // AddAttribute(attrs, "input_padding_top", pads[0]);
  // AddAttribute(attrs, "input_padding_right", pads[3]);
  // AddAttribute(attrs, "input_padding_bottom", pads[2]);
  // AddAttribute(attrs, "input_padding_left", pads[1]);

  // AddAttribute(attrs, "subsampling_height", strides[0]);
  // AddAttribute(attrs, "subsampling_width", strides[1]);

  // AddAttribute(attrs, "dilation_height", dilations[0]);
  // AddAttribute(attrs, "dilation_width", dilations[1]);

Status Convolution2d::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = nullptr;
  if (has_const_output_shape_) {
    Y = context->Output(0, output_shape_);
  } else {
    std::array<int64_t, 4> output_dims;
    Status status = XnnPackConvShapeInferKernelImpl(
        X->Shape(), context->Input<Tensor>(1)->Shape(), input_padding_top_, input_padding_right_, input_padding_bottom_,
        input_padding_left_, subsampling_height_, subsampling_width_, dilation_height_, dilation_width_,
        static_cast<ONNXRUNTIME_XNNPACK_PADDING_MODE>(padding_mode_), output_dims);
    if (!status.IsOK()) {
      return Status(common::ONNXRUNTIME, common::FAIL, status.ErrorMessage());
    }
    Y = context->Output(0, TensorShape(output_dims));
  }

  const TensorShape& input_shape = X->Shape();
  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      op0_.get(), input_shape[0] * batch size *, input_shape[1] * input height *, input_shape[2] * input width *,
      X->Data<float>() * input *, Y->MutableData<float>() * output *, nullptr * threadpool *);
  ORT_ENFORCE(status == xnn_status_success);
  status = xnn_run_operator(op0_.get(), nullptr);
  ORT_ENFORCE(status == xnn_status_success);
  return Status::OK();

  */

  /*
Status DepthWiseConvolution2d::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  Tensor* Y = nullptr;
  if (has_const_output_shape_) {
    Y = context->Output(0, output_shape_);
  } else {
    const ONNX_NAMESPACE::TensorShapeProto* weight_shape = Node().InputDefs()[1]->Shape();
    ORT_ENFORCE(weight_shape != nullptr);
    const ONNX_NAMESPACE::TensorShapeProto input_shape = ToTensorShapeProto(X->Shape());
    ONNX_NAMESPACE::TensorShapeProto final_output_shape;

    OnnxStatus status = XnnPackDepthwiseConvolution2dShapeInferImpl(
        input_shape, *weight_shape, input_padding_top_, input_padding_right_, input_padding_bottom_,
        input_padding_left_, subsampling_height_, subsampling_width_, dilation_height_, dilation_width_,
        static_cast<ONNXRUNTIME_XNNPACK_PADDING_MODE>(padding_mode_), &final_output_shape);
    if (!status.IsOK()) {
      return Status(common::ONNXRUNTIME, common::FAIL, status.ErrorMessage());
    }
    TensorShape output_shape = utils::GetTensorShapeFromTensorShapeProto(final_output_shape);
    if (!IsAllDimKnown(output_shape)) {
      // If it happens, we have a logic error
      return Status(common::ONNXRUNTIME, common::FAIL, "Cannot infer output shape");
    }
    Y = context->Output(0, output_shape);
  }
  const TensorShape& input_shape = X->Shape();
  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      op0_.get(), input_shape[0] * batch size *, input_shape[1] * input height *, input_shape[2] * input width *,
      X->Data<float>() * input *, Y->MutableData<float>() * output *, nullptr * threadpool *);
  ORT_ENFORCE(status == xnn_status_success);
  status = xnn_run_operator(op0_.get(), nullptr);
  ORT_ENFORCE(status == xnn_status_success);

  return Status::OK();
}
  */
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
