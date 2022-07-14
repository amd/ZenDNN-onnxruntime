// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/binary_elementwise_args.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

struct BinaryElementwisePreparation {
  const Tensor* lhs_tensor = nullptr;
  const Tensor* rhs_tensor = nullptr;
  Tensor* output_tensor = nullptr;
  BinaryElementwiseArgs args{};

  BinaryElementwisePreparation() {}
  void BinaryElementwiseBroadcastPrepareHelper(const TensorShape& lhs_shape, const TensorShape& rhs_shape,
                                               const TensorShape& output_shape);
};

void BinaryElementwiseBroadcastPrepare(
    const Tensor* lhs_tensor,
    const Tensor* rhs_tensor,
    Tensor* output_tensor,
    BinaryElementwisePreparation* p,
    const TensorShape* override_lhs_shape = nullptr,
    const TensorShape* override_rhs_shape = nullptr);

// trait classes to indicate if the kernel supports broadcast
class ShouldBroadcast {
};

class ShouldNotBroadcast {
};

template <typename BroadcastTrait>
class BinaryElementwise : public CudaKernel {
 protected:
  typedef BroadcastTrait broadcast_type;

  BinaryElementwise(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  Status Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const;
};

template <typename T>
class Add final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Add(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Sub(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Mul(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Div(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Pow_7 final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Pow_7(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// Since version 12
class Pow final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Pow(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class And final : public BinaryElementwise<ShouldBroadcast> {
 public:
  And(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Or final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Or(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Xor final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Xor(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// PRelu is activation function, but it's closer to binary elementwise ops in implementation
template <typename T>
class PRelu final : public BinaryElementwise<ShouldBroadcast> {
 public:
  PRelu(const OpKernelInfo& info) : BinaryElementwise(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T, typename CudaT>
class CompareFunction : public BinaryElementwise<ShouldBroadcast> {
 public:
  CompareFunction(const OpKernelInfo& info) : BinaryElementwise(info) {}

  typedef void (*ImplCompare)(cudaStream_t stream, const CudaT* lhs_data, const CudaT* rhs_data, bool* output_data,
                              const BinaryElementwiseArgs& args);

  Status CompareMethod(OpKernelContext* context, ImplCompare Impl_Compare) const;
};

template <typename T>
class Greater final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Greater(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Equal final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Equal(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Less final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Less(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GreaterOrEqual final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  GreaterOrEqual(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class LessOrEqual final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  LessOrEqual(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
