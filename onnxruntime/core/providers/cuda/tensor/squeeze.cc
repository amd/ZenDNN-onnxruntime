// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/squeeze.h"

#include "core/providers/cuda/tensor/copy.h"

namespace onnxruntime {
namespace cuda {

#ifdef ENABLE_TRAINING
#define CREATE_SQUEEZE_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedInput(0).MayStridedOutput(0, 0)
#else
#define CREATE_SQUEEZE_KERNEL_DEF (*KernelDefBuilder::Create())
#endif

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze, kOnnxDomain, 1, 10, kCudaExecutionProvider,
    CREATE_SQUEEZE_KERNEL_DEF.Alias(0, 0).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), Squeeze);

// explicit support for negative axis.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze, kOnnxDomain, 11, 12, kCudaExecutionProvider,
    CREATE_SQUEEZE_KERNEL_DEF.Alias(0, 0).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()), Squeeze);

// axes is input instead of attribute
ONNX_OPERATOR_KERNEL_EX(Squeeze, kOnnxDomain, 13, kCudaExecutionProvider,
                        CREATE_SQUEEZE_KERNEL_DEF.Alias(0, 0)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .InputMemoryType(OrtMemTypeCPUInput, 1),
                        Squeeze);

Status Squeeze::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();

  TensorShapeVector axes;
  size_t num_inputs = ctx->InputCount();
  if (num_inputs == 2) {  //axes is an input
    const Tensor* axes_tensor = ctx->Input<Tensor>(1);
    ORT_ENFORCE(axes_tensor != nullptr, "Axes input is null");
    ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1,
                "An axes tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(axes_tensor->Shape()[0]);
    const auto* data = axes_tensor->template Data<int64_t>();
    axes.assign(data, data + nDims);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }

  TensorShapeVector output_shape = ComputeOutputShape(X_shape, axes);

  Tensor* Y = ctx->Output(0, TensorShape(output_shape));

  const void* input = X->DataRaw();
  void* output = Y->MutableDataRaw();
  if (input == output) {
#ifdef ENABLE_TRAINING
    auto original_strides = X->Strides();
    TensorShapeVector new_strides;
    auto num_dimensions = X_shape.NumDimensions();

    // Handle negtive axis, then resort and uniq.
    TensorShapeVector axes_corrected(axes.size());
    for (size_t i = 0; i < axes.size(); i++) {
      axes_corrected[i] = HandleNegativeAxis(axes[i], num_dimensions);
    }
    std::sort(axes_corrected.begin(), axes_corrected.end());
    axes_corrected.erase(std::unique(axes_corrected.begin(), axes_corrected.end()), axes_corrected.end());

    size_t j = 0;
    for (size_t i = 0; i < num_dimensions; ++i) {
      if ((j < axes_corrected.size() && axes_corrected[j] == static_cast<int64_t>(i)) ||
          (axes_corrected.size() == 0 && X->Shape()[i] == 1)) {
        ++j;
        continue;
      }
      new_strides.emplace_back(original_strides[i]);
    }
    Y->SetShapeAndStrides(TensorShape(output_shape), new_strides);
#endif  // ENABLE_TRAINING
    return Status::OK();
  }

  return StridedCopyTensor(Stream(), *X, *Y);
}

}  // namespace cuda
}  // namespace onnxruntime
