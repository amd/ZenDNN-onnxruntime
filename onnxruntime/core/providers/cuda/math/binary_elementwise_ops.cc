// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/binary_elementwise_ops.h"

#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/elementwise_utils.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

namespace {

static bool TryPerChannel(size_t rank, const TensorShapeVector& input_strides, const TensorShapeVector& output_shapes,
                          bool& is_multi_batch, int& channel, int& height) {
  size_t count = 0;
  size_t dim_channel = 0;
  for (size_t dim = 0; dim < rank; ++dim) {
    if (output_shapes[dim] != 1 && input_strides[dim] != 0) {
      if (count > 0) return false;
      ++count;
      channel = output_shapes[dim];
      dim_channel = dim;
    }
  }
  if (count == 0) return false;
  is_multi_batch = false;
  for (size_t dim = 0; dim < dim_channel; ++dim) {
    if (output_shapes[dim] > 1) {
      is_multi_batch = true;
      break;
    }
  }
  height = 1;
  for (size_t dim = dim_channel + 1; dim < rank; ++dim) {
    height *= static_cast<int>(output_shapes[dim]);
  }
  return true;
}

}  // namespace

void BinaryElementwisePreparation::BinaryElementwiseBroadcastPrepareHelper(const TensorShape& lhs_shape,
                                                                           const TensorShape& rhs_shape,
                                                                           const TensorShape& output_shape) {
  bool lhs_is_contiguous = true;
  bool rhs_is_contiguous = true;
  TensorShapeVector lhs_original_strides = TensorPitches(lhs_shape);
  TensorShapeVector rhs_original_strides = TensorPitches(rhs_shape);
#ifdef ENABLE_TRAINING
  if (lhs_tensor) {
    lhs_is_contiguous = lhs_tensor->IsContiguous();
    if (!lhs_is_contiguous) {
      ORT_ENFORCE(lhs_tensor->Shape() == lhs_shape);
      lhs_original_strides = ToShapeVector(lhs_tensor->Strides());
    }
  }
  if (rhs_tensor) {
    rhs_is_contiguous = rhs_tensor->IsContiguous();
    if (!rhs_is_contiguous) {
      ORT_ENFORCE(rhs_tensor->Shape() == rhs_shape);
      rhs_original_strides = ToShapeVector(rhs_tensor->Strides());
    }
  }
#endif
  args.lhs_index_type = lhs_shape.Size() == 1                            ? BroadcastIndexType::Scalar
                        : lhs_is_contiguous && lhs_shape == output_shape ? BroadcastIndexType::NoBroadcast
                                                                         : BroadcastIndexType::NeedCompute;
  args.rhs_index_type = rhs_shape.Size() == 1                            ? BroadcastIndexType::Scalar
                        : rhs_is_contiguous && rhs_shape == output_shape ? BroadcastIndexType::NoBroadcast
                                                                         : BroadcastIndexType::NeedCompute;
  args.output_size = static_cast<size_t>(output_shape.Size());
  // Reset per_channel_type in case this preparation is reused.
  args.per_channel_type = PerChannelType::None;

  // Other args are not needed if none of sizes needs compute.
  if (args.lhs_index_type != BroadcastIndexType::NeedCompute &&
      args.rhs_index_type != BroadcastIndexType::NeedCompute) {
    return;
  }

  size_t rank = output_shape.NumDimensions();
  TensorShapeVector output_dims = output_shape.AsShapeVector();
  size_t lhs_offset = rank - lhs_original_strides.size();
  size_t rhs_offset = rank - rhs_original_strides.size();
  TensorShapeVector lhs_strides(rank, 0);
  for (size_t i = 0; i < lhs_original_strides.size(); ++i) {
    lhs_strides[lhs_offset + i] = (lhs_shape[i] == 1 && output_dims[lhs_offset + i] != 1) ? 0 : lhs_original_strides[i];
  }
  TensorShapeVector rhs_strides(rank, 0);
  for (size_t i = 0; i < rhs_original_strides.size(); ++i) {
    rhs_strides[rhs_offset + i] = (rhs_shape[i] == 1 && output_dims[rhs_offset + i] != 1) ? 0 : rhs_original_strides[i];
  }

  CoalesceDimensions({lhs_strides, rhs_strides}, output_dims);
  rank = output_dims.size();
  args.rank = rank;

  // special case for lhs(N,C,H) and rhs (C,1) which is used in conv bias
  // when N == 1: out[id] = op(lhs[id], rhs[id / H])
  // When N > 1:  out[id] = op(lhs[id], rhs[id / H % C])
  bool is_multi_batch;
  int channel, height;
  if (args.lhs_index_type == BroadcastIndexType::NoBroadcast &&
      args.rhs_index_type == BroadcastIndexType::NeedCompute &&
      TryPerChannel(rank, rhs_strides, output_dims, is_multi_batch, channel, height)) {
    args.per_channel_type = PerChannelType::RhsNeedCompute;
    args.is_multi_batch = is_multi_batch;
    args.channel = channel;
    args.height = height;
  } else if (args.lhs_index_type == BroadcastIndexType::NeedCompute &&
             args.rhs_index_type == BroadcastIndexType::NoBroadcast &&
             TryPerChannel(rank, lhs_strides, output_dims, is_multi_batch, channel, height)) {
    args.per_channel_type = PerChannelType::LhsNeedCompute;
    args.is_multi_batch = is_multi_batch;
    args.channel = channel;
    args.height = height;
  } else {
    args.output_fdms.SetSize(static_cast<int>(rank));
    TensorPitches output_strides(output_dims);
    for (int i = 0; i < static_cast<int>(rank); ++i) {
      args.output_fdms[i] = fast_divmod(static_cast<int>(output_strides[i]));
    }
    if (args.lhs_index_type == BroadcastIndexType::NeedCompute) args.lhs_strides = TArray<int64_t>(lhs_strides);
    if (args.rhs_index_type == BroadcastIndexType::NeedCompute) args.rhs_strides = TArray<int64_t>(rhs_strides);
  }
}

template <>
Status BinaryElementwise<ShouldNotBroadcast>::Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const {
  p->lhs_tensor = context->Input<Tensor>(0);
  p->rhs_tensor = context->Input<Tensor>(1);
  if (!(p->lhs_tensor->Shape() == p->rhs_tensor->Shape()))
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, Node().Name(), ": mismatching input shapes: ",
                           p->lhs_tensor->Shape().ToString(), " != ", p->rhs_tensor->Shape().ToString());
  p->output_tensor = context->Output(0, p->lhs_tensor->Shape());
  p->args.lhs_index_type = BroadcastIndexType::NoBroadcast;
  p->args.rhs_index_type = BroadcastIndexType::NoBroadcast;
  p->args.output_size = static_cast<size_t>(p->output_tensor->Shape().Size());
  return Status::OK();
}

void BinaryElementwiseBroadcastPrepare(const Tensor* lhs_tensor, const Tensor* rhs_tensor, Tensor* output_tensor,
                                       BinaryElementwisePreparation* p, const TensorShape* override_lhs_shape,
                                       const TensorShape* override_rhs_shape) {
  p->lhs_tensor = lhs_tensor;
  p->rhs_tensor = rhs_tensor;
  const auto& lhs_shape = override_lhs_shape ? *override_lhs_shape : lhs_tensor->Shape();
  const auto& rhs_shape = override_rhs_shape ? *override_rhs_shape : rhs_tensor->Shape();
  p->output_tensor = output_tensor;
  const auto& output_shape = output_tensor->Shape();
  p->BinaryElementwiseBroadcastPrepareHelper(lhs_shape, rhs_shape, output_shape);
}

template <>
Status BinaryElementwise<ShouldBroadcast>::Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const {
  auto lhs_tensor = context->Input<Tensor>(0);
  auto rhs_tensor = context->Input<Tensor>(1);
  const auto& lhs_shape = lhs_tensor->Shape();
  const auto& rhs_shape = rhs_tensor->Shape();
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), {lhs_shape, rhs_shape}, output_shape));
  auto output_tensor = context->Output(0, output_shape);
  BinaryElementwiseBroadcastPrepare(lhs_tensor, rhs_tensor, output_tensor, p);
  return Status::OK();
}

#ifdef ENABLE_TRAINING
#define CREATE_BINARY_ELEMENTWISE_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedInput(0).MayStridedInput(1)
#else
#define CREATE_BINARY_ELEMENTWISE_KERNEL_DEF (*KernelDefBuilder::Create())
#endif

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_V(x, class_name, ver, T) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                          \
      x, kOnnxDomain, ver, T, kCudaExecutionProvider,                     \
      CREATE_BINARY_ELEMENTWISE_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), class_name<T>);

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(x, ver, T) BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED_V(x, x, ver, T)

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_NONTEMP(x, class_name, ver, ...)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                           \
      x, kOnnxDomain, ver, kCudaExecutionProvider,                                                         \
      CREATE_BINARY_ELEMENTWISE_KERNEL_DEF.TypeConstraint("T", BuildKernelDefConstraints<>(__VAR_ARGS__)), \
      class_name);

#define BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(x, ver, T)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                 \
      x, kOnnxDomain, ver, T, kCudaExecutionProvider,                                            \
      CREATE_BINARY_ELEMENTWISE_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),                            \
      x<T>);

#define BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(x, startver, endver, T)     \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                       \
      x, kOnnxDomain, startver, endver, T, kCudaExecutionProvider,                               \
      CREATE_BINARY_ELEMENTWISE_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),                            \
      x<T>);

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(x, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                         \
      x, kOnnxDomain, startver, endver, T, kCudaExecutionProvider,                 \
      CREATE_BINARY_ELEMENTWISE_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), x<T>);

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED_CLASS(x, class_name, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                           \
      x, kOnnxDomain, startver, endver, T, kCudaExecutionProvider,                                   \
      CREATE_BINARY_ELEMENTWISE_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), class_name<T>);

#define BINARY_ELEMENTWISE_COMPUTE(x, T)                                                                               \
  template <>                                                                                                          \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                                       \
    BinaryElementwisePreparation prepare;                                                                              \
    ORT_RETURN_IF_ERROR(Prepare(context, &prepare));                                                                   \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                                      \
        Stream(), reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()), \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),           \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),       \
        prepare.args);                                                                                                 \
    return Status::OK();                                                                                               \
  }

#define BINARY_OP_VERSIONED_TYPED(name, startver, endver, T) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, T)

#define BINARY_OP_TYPED(name, ver, T)                    \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, T) \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

#define BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, T)                        \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED_CLASS(name, class_name, startver, endver, T) \
  BINARY_ELEMENTWISE_COMPUTE(class_name, T)

#define BINARY_LOGICALOP_TYPED(name, ver, T)                       \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, T) \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

// since different ops has different types, we cannot use BINARY_OPS() directly
// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

#define BINARY_OP_VERSIONED_HFD(name, startver, endver)        \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, float)     \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, double)

#define BINARY_OP_VERSIONED_UZILHFD(name, startver, endver)   \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, uint32_t) \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, uint64_t) \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int32_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int64_t)  \
  BINARY_OP_VERSIONED_HFD(name, startver, endver)

#define BINARY_OP_VERSIONED_UZILHFD_WITH_BF16(name, startver, endver) \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, uint32_t)         \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, uint64_t)         \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int32_t)          \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int64_t)          \
  BINARY_OP_VERSIONED_HFD(name, startver, endver)                     \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, BFloat16)

#define BINARY_OP_HFD(name, ver)        \
  BINARY_OP_TYPED(name, ver, MLFloat16) \
  BINARY_OP_TYPED(name, ver, float)     \
  BINARY_OP_TYPED(name, ver, double)    \
  BINARY_OP_TYPED(name, ver, BFloat16)

#define BINARY_OP_UZILHFD(name, ver)   \
  BINARY_OP_TYPED(name, ver, uint32_t) \
  BINARY_OP_TYPED(name, ver, uint64_t) \
  BINARY_OP_TYPED(name, ver, int32_t)  \
  BINARY_OP_TYPED(name, ver, int64_t)  \
  BINARY_OP_HFD(name, ver)

#define BINARY_OP_REGISTER_VERSIONED_OIL(name, startver, endver)                      \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, bool)    \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int32_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int64_t)

#define BINARY_LOGICALOP_REGISTER_OIL(name, ver)                         \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, bool)    \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, int32_t) \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, int64_t)

#define BINARY_OP_REGISTER_HFD(name, ver)                        \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, MLFloat16) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, float)     \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, double)    \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, BFloat16)

#define BINARY_OP_REGISTER_UZILHFD(name, ver)                   \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, uint32_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, uint64_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, int32_t)  \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, int64_t)  \
  BINARY_OP_REGISTER_HFD(name, ver)

#define BINARY_LOGICALOP_REGISTER_UZILHFD(name, ver)                       \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, uint32_t)  \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, uint64_t)  \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, int32_t)   \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, int64_t)   \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, MLFloat16) \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, float)     \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, double)    \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(name, ver, BFloat16)

#define BINARY_LOGICALOP_REGISTER_VERSIONED_UZILHFD(name, startver, endver)                       \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, uint32_t)  \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, uint64_t)  \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int32_t)   \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int64_t)   \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, float)     \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, double)    \
  BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, BFloat16)

#define BINARY_OP_REGISTER_VERSIONED_HFD(name, startver, endver)                        \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, float)     \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, double)    \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, BFloat16)

#define BINARY_OP_REGISTER_VERSIONED_CLASS_HFD(name, class_name, startver, endver) \
  BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, MLFloat16)       \
  BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, float)           \
  BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, double)          \
  BINARY_OP_TYPED_VERSIONED_V(name, class_name, startver, endver, BFloat16)

#define BINARY_OP_REGISTER_VERSIONED_UZILHFD(name, startver, endver)                   \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, uint32_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, uint64_t) \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int32_t)  \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, int64_t)  \
  BINARY_OP_REGISTER_VERSIONED_HFD(name, startver, endver)

BINARY_OP_VERSIONED_UZILHFD(Add, 7, 12)
BINARY_OP_VERSIONED_UZILHFD(Sub, 7, 12)
BINARY_OP_VERSIONED_UZILHFD(Mul, 7, 12)
BINARY_OP_VERSIONED_UZILHFD(Div, 7, 12)

BINARY_OP_VERSIONED_UZILHFD_WITH_BF16(Add, 13, 13)
BINARY_OP_VERSIONED_UZILHFD_WITH_BF16(Sub, 13, 13)
BINARY_OP_VERSIONED_UZILHFD_WITH_BF16(Mul, 13, 13)
BINARY_OP_VERSIONED_UZILHFD_WITH_BF16(Div, 13, 13)

BINARY_OP_UZILHFD(Add, 14)
BINARY_OP_UZILHFD(Sub, 14)
BINARY_OP_UZILHFD(Mul, 14)
BINARY_OP_UZILHFD(Div, 14)

BINARY_OP_REGISTER_VERSIONED_CLASS_HFD(Pow, Pow_7, 7, 11)
BINARY_LOGICALOP_TYPED(And, 7, bool)
BINARY_LOGICALOP_TYPED(Or, 7, bool)
BINARY_LOGICALOP_TYPED(Xor, 7, bool)
BINARY_OP_VERSIONED_HFD(PRelu, 7, 8)
BINARY_OP_VERSIONED_HFD(PRelu, 9, 15)
// Opset-16 adds BFloat16 to allowed types for the PRelu operator
BINARY_OP_HFD(PRelu, 16)

// Pow since version 12
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pow, kOnnxDomain, 12, 12, kCudaExecutionProvider,
    CREATE_BINARY_ELEMENTWISE_KERNEL_DEF
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>()),
    Pow);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Pow, kOnnxDomain, 13, 14, kCudaExecutionProvider,
    CREATE_BINARY_ELEMENTWISE_KERNEL_DEF
        .TypeConstraint("T", BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>()),
    Pow);

ONNX_OPERATOR_KERNEL_EX(Pow, kOnnxDomain, 15, kCudaExecutionProvider,
                        CREATE_BINARY_ELEMENTWISE_KERNEL_DEF
                            .TypeConstraint("T",
                                            BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>())
                            .TypeConstraint("T1",
                                            BuildKernelDefConstraints<int32_t, int64_t, float, double, MLFloat16>()),
                        Pow);

#define TYPED_IMPLT1_POW(rhs_onnx_type, rhs_data_type)                                                               \
  case rhs_onnx_type: {                                                                                              \
    ImplT1_Pow<typename ToCudaType<T>::MappedType, typename ToCudaType<rhs_data_type>::MappedType>(                  \
        stream, reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()), \
        reinterpret_cast<const typename ToCudaType<rhs_data_type>::MappedType*>(                                     \
            prepare.rhs_tensor->template Data<rhs_data_type>()),                                                     \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),     \
        prepare.args);                                                                                               \
  } break

namespace pow12_internal {
template <class T>
Status DispatchOnFirstArg(cudaStream_t stream, const BinaryElementwisePreparation& prepare) {
  namespace on = ONNX_NAMESPACE;
  Status s;
  switch (prepare.rhs_tensor->GetElementType()) {
#define CASE_IMPLT1_POW_RHL_TYPE(rhs_onnx_type, rhs_data_type)                                                       \
  case rhs_onnx_type: {                                                                                              \
    ImplT1_Pow<typename ToCudaType<T>::MappedType, typename ToCudaType<rhs_data_type>::MappedType>(                  \
        stream, reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()), \
        reinterpret_cast<const typename ToCudaType<rhs_data_type>::MappedType*>(                                     \
            prepare.rhs_tensor->template Data<rhs_data_type>()),                                                     \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()),     \
        prepare.args);                                                                                               \
  } break
    CASE_IMPLT1_POW_RHL_TYPE(on::TensorProto_DataType_INT32, int32_t);
    CASE_IMPLT1_POW_RHL_TYPE(on::TensorProto_DataType_INT64, int64_t);
    CASE_IMPLT1_POW_RHL_TYPE(on::TensorProto_DataType_FLOAT, float);
    CASE_IMPLT1_POW_RHL_TYPE(on::TensorProto_DataType_DOUBLE, double);
    CASE_IMPLT1_POW_RHL_TYPE(on::TensorProto_DataType_FLOAT16, MLFloat16);
#undef CASE_IMPLT1_POW_RHL_TYPE
    default:
      s = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                          "Unsupported Y type: ", DataTypeImpl::ToString(prepare.rhs_tensor->DataType()));
  }
  return s;
}
}  // namespace pow12_internal

Status Pow::ComputeInternal(OpKernelContext* context) const {
  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(Prepare(context, &prepare));
  namespace on = ONNX_NAMESPACE;
  using namespace pow12_internal;

  Status s;

  switch (prepare.lhs_tensor->GetElementType()) {
    case on::TensorProto_DataType_INT32:
      s = DispatchOnFirstArg<int32_t>(Stream(), prepare);
      break;
    case on::TensorProto_DataType_INT64:
      s = DispatchOnFirstArg<int64_t>(Stream(), prepare);
      break;
    case on::TensorProto_DataType_FLOAT:
      s = DispatchOnFirstArg<float>(Stream(), prepare);
      break;
    case on::TensorProto_DataType_DOUBLE:
      s = DispatchOnFirstArg<double>(Stream(), prepare);
      break;
    case on::TensorProto_DataType_FLOAT16:
      s = DispatchOnFirstArg<MLFloat16>(Stream(), prepare);
      break;
    default:
      s = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported X type: ",
                          DataTypeImpl::ToString(prepare.lhs_tensor->DataType()));
  }
  return s;
}

//Greater op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T, typename CudaT>
Status CompareFunction<T, CudaT>::CompareMethod(OpKernelContext* context, ImplCompare Impl_Compare) const {
  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(Prepare(context, &prepare));

  Impl_Compare(Stream(), reinterpret_cast<const CudaT*>(prepare.lhs_tensor->template Data<T>()),
               reinterpret_cast<const CudaT*>(prepare.rhs_tensor->template Data<T>()),
               reinterpret_cast<ToCudaType<bool>::MappedType*>(prepare.output_tensor->template MutableData<bool>()),
               prepare.args);

  return Status::OK();
}

//Greater op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status Greater<T>::ComputeInternal(OpKernelContext* context) const {
  return this->CompareMethod(context, &ImplT2_Greater);
}

template <typename T>
Status Equal<T>::ComputeInternal(OpKernelContext* context) const {
  return this->CompareMethod(context, &ImplT2_Equal);
}

//Less op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status Less<T>::ComputeInternal(OpKernelContext* context) const {
  return this->CompareMethod(context, &ImplT2_Less);
}

//GreaterOrEqual op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status GreaterOrEqual<T>::ComputeInternal(OpKernelContext* context) const {
  return this->CompareMethod(context, &ImplT2_GreaterOrEqual);
}

//LessOrEqual op output tensor type is bool, so it cannot directly fit in the macros
//for other elementwise ops
template <typename T>
Status LessOrEqual<T>::ComputeInternal(OpKernelContext* context) const {
  return this->CompareMethod(context, &ImplT2_LessOrEqual);
}

BINARY_LOGICALOP_REGISTER_UZILHFD(Equal, 13)
BINARY_ELEMENTWISE_LOGICALOP_REGISTER_KERNEL_TYPED(Equal, 13, bool)
BINARY_OP_REGISTER_VERSIONED_UZILHFD(Equal, 11, 12)
BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(Equal, 11, 12, bool)
BINARY_OP_REGISTER_VERSIONED_OIL(Equal, 7, 10)
BINARY_LOGICALOP_REGISTER_UZILHFD(Greater, 13)
BINARY_OP_REGISTER_VERSIONED_UZILHFD(Greater, 9, 12)
BINARY_OP_REGISTER_VERSIONED_HFD(Greater, 7, 8)
BINARY_LOGICALOP_REGISTER_UZILHFD(Less, 13)
BINARY_OP_REGISTER_VERSIONED_UZILHFD(Less, 9, 12)
BINARY_OP_REGISTER_VERSIONED_HFD(Less, 7, 8)
BINARY_LOGICALOP_REGISTER_VERSIONED_UZILHFD(GreaterOrEqual, 12, 15)
BINARY_LOGICALOP_REGISTER_VERSIONED_UZILHFD(LessOrEqual, 12, 15)

// Opset-16 adds BFloat16 to allowed types for the GreaterOrEqual operator
BINARY_LOGICALOP_REGISTER_UZILHFD(GreaterOrEqual, 16)

// Opset-16 adds BFloat16 to allowed types for the LessOrEqual operator
BINARY_LOGICALOP_REGISTER_UZILHFD(LessOrEqual, 16)

}  // namespace cuda
}  // namespace onnxruntime
