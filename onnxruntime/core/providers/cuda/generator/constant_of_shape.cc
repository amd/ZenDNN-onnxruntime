// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "constant_of_shape.h"

using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace cuda {

#ifdef ENABLE_TRAINING
#define CREATE_CONST_OF_SHAPE_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedOutput(0, -1)
#else
#define CREATE_CONST_OF_SHAPE_KERNEL_DEF (*KernelDefBuilder::Create())
#endif

ONNX_OPERATOR_KERNEL_EX(ConstantOfShape, kOnnxDomain, 9, kCudaExecutionProvider,
                        CREATE_CONST_OF_SHAPE_KERNEL_DEF.InputMemoryType(OrtMemTypeCPUInput, 0)
                            .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorTypes()),
                        ConstantOfShape);

Status ConstantOfShape::ComputeInternal(OpKernelContext* ctx) const {
  Tensor* output_tensor = nullptr;
  bool is_strided = false;
#ifdef ENABLE_TRAINING
  is_strided = ctx->IsStridedTensor(0);
#endif  // ENABLE_TRAINING
  ORT_RETURN_IF_ERROR(PrepareCompute(ctx, &output_tensor, is_strided));
  auto output_data = output_tensor->MutableDataRaw();
  const auto size = is_strided ? 1 : output_tensor->Shape().Size();
  const void* value_ptr = GetValuePtr();
  const auto element_size = output_tensor->DataType()->Size();

#define CASE(TYPE)                                                                                                   \
  case sizeof(TYPE):                                                                                                 \
    if (size > 0) {                                                                                                  \
      cuda::Fill(Stream(), reinterpret_cast<TYPE*>(output_data), *(reinterpret_cast<const TYPE*>(value_ptr)), size); \
    }                                                                                                                \
    break;

  switch (element_size) {
    CASE(int8_t)
    CASE(int16_t)
    CASE(int32_t)
    CASE(int64_t)
    default:
      ORT_THROW("Unsupported value attribute datatype with sizeof=: ", element_size);
      break;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
