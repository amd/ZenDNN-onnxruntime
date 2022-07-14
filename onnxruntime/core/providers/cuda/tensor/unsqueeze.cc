// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/unsqueeze.h"

#include "core/providers/cuda/tensor/copy.h"

namespace onnxruntime {
namespace cuda {

#ifdef ENABLE_TRAINING
#define CREATE_UNSQUEEZE_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedInput(0).MayStridedOutput(0, 0)
#else
#define CREATE_UNSQUEEZE_KERNEL_DEF (*KernelDefBuilder::Create())
#endif

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze, kOnnxDomain, 1, 10, kCudaExecutionProvider,
    CREATE_UNSQUEEZE_KERNEL_DEF.Alias(0, 0).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), Unsqueeze);

// explicitly support negative axis
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Unsqueeze, kOnnxDomain, 11, 12, kCudaExecutionProvider,
    CREATE_UNSQUEEZE_KERNEL_DEF.Alias(0, 0).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), Unsqueeze);

// axes is input instead of attribute, support bfloat16
ONNX_OPERATOR_KERNEL_EX(Unsqueeze, kOnnxDomain, 13, kCudaExecutionProvider,
                        CREATE_UNSQUEEZE_KERNEL_DEF.Alias(0, 0)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .InputMemoryType(OrtMemTypeCPUInput, 1),
                        Unsqueeze);

Status Unsqueeze::ComputeInternal(OpKernelContext* ctx) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, p));

  const void* input = p.input_tensor->DataRaw();
  void* output = p.output_tensor->MutableDataRaw();
  if (input == output) {
#ifdef ENABLE_TRAINING
    TensorShapeVector axes;
    size_t num_inputs = ctx->InputCount();
    if (num_inputs == 2) {  // axes is an input
      const Tensor* axes_tensor = ctx->Input<Tensor>(1);
      ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
      ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 0 || axes_tensor->Shape().NumDimensions() == 1,
                  "An axes tensor must be a scalar or a 1-D tensor.");
      auto data_span = axes_tensor->template DataAsSpan<int64_t>();
      axes.assign(data_span.cbegin(), data_span.cend());
    } else {
      axes.assign(axes_.begin(), axes_.end());
    }

    TensorShapeVector output_strides(axes.size() + p.input_tensor->Shape().NumDimensions(), -1);
    for (int64_t axis : axes) {
      axis = HandleNegativeAxis(axis, output_strides.size());
      output_strides[axis] = 0;
    }

    auto original_strides = p.input_tensor->Strides();
    for (size_t i = 0, j = 0; i < output_strides.size(); ++i) {
      if (output_strides[i] == 0) continue;
      output_strides[i] = original_strides[j++];
    }
    p.output_tensor->SetShapeAndStrides(p.output_tensor->Shape(), output_strides);
#endif  // ENABLE_TRAINING
    return Status::OK();
  }

  return StridedCopyTensor(Stream(), *p.input_tensor, *p.output_tensor);
}

}  // namespace cuda
}  // namespace onnxruntime
