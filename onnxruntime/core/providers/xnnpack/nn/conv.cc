// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"
#include "core/graph/constants.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"

namespace onnxruntime {
namespace xnnpack {

Conv::Conv(const OpKernelInfo& info) : OpKernel(info) {
  std::string activation;
  if (info.GetAttr<std::string>("activation", &activation).IsOK()) {
    std::vector<float> activation_params;

    if (info.GetAttrs<float>("activation_params", activation_params).IsOK()) {
      if (activation_params.size() == 2) {
        clip_min_max_ = {activation_params[0],
                         activation_params[1]};
      }
    }
  }
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

    // NOTE: For this demo create a Tensor for the packed weight so there's a shape attached.
    //       Could alternatively create a raw buffer with IAllocator::MakeUniquePtr<void> if the overhead of a Tensor
    //       is not needed.

    // arbitrary example moving first dim to the end.
    // in the real implementation the transpose of the data would also be done.
    std::vector<int64_t> new_shape;
    auto rank = orig_shape.NumDimensions();
    new_shape.reserve(rank);

    for (size_t i = 1; i < rank; ++i) {
      new_shape.push_back(orig_shape[i]);
    }

    new_shape.push_back(orig_shape[0]);

    packed_w_ = Tensor::Create(tensor.DataType(), TensorShape(new_shape), alloc);

    // set to arbitrary value
    memset(packed_w_->MutableDataRaw(), 7, packed_w_->SizeInBytes());

    is_packed = true;
  }

  return Status::OK();
}

Status Conv::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // this is in NHWC format
  // const Tensor& W = *packed_w_;
  // ...
  // return Status::OK();
  std::cout << "Compute called with input shape of " << X.Shape() << "\n";
  ORT_NOT_IMPLEMENTED("TODO: add NHWC implementation here.");
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
