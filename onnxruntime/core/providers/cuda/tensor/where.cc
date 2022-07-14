// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/where.h"

#include "core/providers/cuda/tensor/where_impl.h"
#include "core/providers/cuda/math/elementwise_utils.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

// kernel builder functions
#define WHERE_TYPED_KERNEL_WITH_TYPE_NAME(T, TName)                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                          \
      Where,                                                        \
      kOnnxDomain,                                                  \
      9,                                                            \
      15,                                                           \
      TName,                                                        \
      kCudaExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                 \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
      Where<T>);                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      Where,                                                        \
      kOnnxDomain,                                                  \
      16,                                                           \
      TName,                                                        \
      kCudaExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                 \
          .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()) \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
      Where<T>);

struct TernaryElementwisePreparation {
  const Tensor* a_tensor = nullptr;
  const Tensor* b_tensor = nullptr;
  const Tensor* c_tensor = nullptr;
  size_t output_rank_or_simple_broadcast = 0;  // for no_broadcast cases, output_rank uses SimpleBroadcast enums
  TArray<int64_t> a_padded_strides;            // for a shape == output shape, this is nullptr
  TArray<int64_t> b_padded_strides;            // for b shape == output shape, this is nullptr
  TArray<int64_t> c_padded_strides;            // for c shape == output shape, this is nullptr
  TArray<fast_divmod> fdm_output_strides;
  BroadcastIndexType a_index_type = BroadcastIndexType::NoBroadcast;
  BroadcastIndexType b_index_type = BroadcastIndexType::NoBroadcast;
  BroadcastIndexType c_index_type = BroadcastIndexType::NoBroadcast;

  TernaryElementwisePreparation(const Tensor* a, const Tensor* b, const Tensor* c)
      : a_tensor(a), b_tensor(b), c_tensor(c) {}

  Status TernaryElementwiseBroadcastPrepareHelper(const TensorShape& a_shape,
                                                  const TensorShape& b_shape,
                                                  const TensorShape& c_shape,
                                                  const TensorShape& output_shape) {
    int32_t a_rank = static_cast<int32_t>(a_shape.NumDimensions());
    int32_t b_rank = static_cast<int32_t>(b_shape.NumDimensions());
    int32_t c_rank = static_cast<int32_t>(c_shape.NumDimensions());
    int32_t out_rank = std::max(std::max(a_rank, b_rank), c_rank);

    // early return when shapes match
    if (a_shape == b_shape && b_shape == c_shape) {
      output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::NoBroadcast);
      return Status::OK();
    }

    output_rank_or_simple_broadcast = out_rank;

    auto padder = [out_rank](int32_t rank, const TensorShape& shape, TArray<int64_t>& padded_strides) {
      padded_strides.SetSize(out_rank);
      if (rank > 0) {
        TensorPitches pitches(shape.GetDims());
        auto offset = out_rank - rank;
        for (auto i = offset; i < out_rank; ++i) {
          // the stride for broadcast dimension is kept as 0
          if (shape.GetDims()[i - offset] != 1) {
            padded_strides[i] = pitches[i - offset];
          }
        }
      }
    };

    bool has_need_compute = false;
    if (a_shape.Size() == 1) {
      a_index_type = BroadcastIndexType::Scalar;
    } else if (a_shape != output_shape) {
      padder(a_rank, a_shape, a_padded_strides);
      a_index_type = BroadcastIndexType::NeedCompute;
      has_need_compute = true;
    }

    if (b_shape.Size() == 1) {
      b_index_type = BroadcastIndexType::Scalar;
    } else if (b_shape != output_shape) {
      padder(b_rank, b_shape, b_padded_strides);
      b_index_type = BroadcastIndexType::NeedCompute;
      has_need_compute = true;
    }

    if (c_shape.Size() == 1) {
      c_index_type = BroadcastIndexType::Scalar;
    } else if (c_shape != output_shape) {
      padder(c_rank, c_shape, c_padded_strides);
      c_index_type = BroadcastIndexType::NeedCompute;
      has_need_compute = true;
    }

    if (!has_need_compute) {
      output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::NoBroadcast);
      return Status::OK();
    }

    TensorPitches output_pitches(output_shape.GetDims());
    fdm_output_strides.SetSize(out_rank);
    for (auto i = 0; i < out_rank; ++i) {
      fdm_output_strides[i] = fast_divmod(static_cast<int32_t>(output_pitches[i]));
    }

    return Status::OK();
  }
};

template <typename T>
Status Where<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto* const condition = context->Input<Tensor>(0);
  const auto* const X = context->Input<Tensor>(1);
  const auto* const Y = context->Input<Tensor>(2);
  ORT_ENFORCE(condition && X && Y, "condition, X, and Y inputs are required!");

  auto const& condition_shape = condition->Shape();
  auto const& X_shape = X->Shape();
  auto const& Y_shape = Y->Shape();

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), {condition_shape, X_shape, Y_shape}, output_shape));
  auto output_tensor = context->Output(0, output_shape);

  if (output_shape.Size() == 0)
    return Status::OK();

  TernaryElementwisePreparation prepare(condition, X, Y);
  ORT_RETURN_IF_ERROR(prepare.TernaryElementwiseBroadcastPrepareHelper(condition_shape, X_shape, Y_shape, output_shape));

  WhereImpl<CudaT>(
      Stream(),
      prepare.output_rank_or_simple_broadcast,
      prepare.a_index_type,
      prepare.a_padded_strides,
      reinterpret_cast<const bool*>(prepare.a_tensor->template Data<bool>()),
      prepare.b_index_type,
      prepare.b_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.b_tensor->template Data<T>()),
      prepare.c_index_type,
      prepare.c_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.c_tensor->template Data<T>()),
      prepare.fdm_output_strides,
      reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>()),
      output_tensor->Shape().Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE_WITH_NAME(T, TName) \
  WHERE_TYPED_KERNEL_WITH_TYPE_NAME(T, TName)   \
  template Status Where<T>::ComputeInternal(OpKernelContext* context) const;

#define SPECIALIZED_COMPUTE(T) \
  SPECIALIZED_COMPUTE_WITH_NAME(T, T)

SPECIALIZED_COMPUTE(uint8_t)
SPECIALIZED_COMPUTE(int32_t)
SPECIALIZED_COMPUTE(int64_t)
SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double_t)
SPECIALIZED_COMPUTE(MLFloat16)
}  // namespace cuda
}  // namespace onnxruntime
