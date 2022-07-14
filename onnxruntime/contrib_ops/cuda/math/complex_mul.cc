// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/complex_mul.h"

#include "contrib_ops/cuda/math/complex_mul_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/elementwise_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ComplexMul,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ComplexMul<T, false>);                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ComplexMulConj,                                             \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ComplexMul<T, true>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T, bool IsConj>
Status ComplexMul<T, IsConj>::ComputeInternal(OpKernelContext* context) const {
  auto lhs_tensor = context->Input<Tensor>(0);
  auto rhs_tensor = context->Input<Tensor>(1);
  const auto& lhs_shape = lhs_tensor->Shape();
  const auto& rhs_shape = rhs_tensor->Shape();
  ORT_ENFORCE(lhs_shape[lhs_shape.NumDimensions() - 1] == 2 && rhs_shape[rhs_shape.NumDimensions() - 1] == 2,
              "Input's last dimension is supposed to be 2.");
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), {lhs_shape, rhs_shape}, output_shape));
  auto output_tensor = context->Output(0, output_shape);

  TensorShapeVector lhs_shape_vec = lhs_shape.AsShapeVector();
  lhs_shape_vec.pop_back();
  TensorShapeVector rhs_shape_vec = rhs_shape.AsShapeVector();
  rhs_shape_vec.pop_back();
  TensorShapeVector output_shape_vec = output_shape.AsShapeVector();
  output_shape_vec.pop_back();
  BinaryElementwisePreparation p;
  p.BinaryElementwiseBroadcastPrepareHelper(TensorShape(lhs_shape_vec), TensorShape(rhs_shape_vec),
                                            TensorShape(output_shape_vec));
  ComplexMul_Impl<typename ToCudaType<T>::MappedType, IsConj>(
      Stream(), reinterpret_cast<const typename ToCudaType<T>::MappedType*>(lhs_tensor->template Data<T>()),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(rhs_tensor->template Data<T>()),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_tensor->template MutableData<T>()), p.args);
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
