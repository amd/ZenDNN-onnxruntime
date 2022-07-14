// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/binary_elementwise_args.h"

namespace onnxruntime {
namespace cuda {

#ifdef USE_ROCM
constexpr int kElementsPerThread = 2;
constexpr int kThreadsPerBlock = 512;
#else
constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
#endif

template <BroadcastIndexType LhsIndexType, BroadcastIndexType RhsIndexType>
struct BinaryOffsetCalculator {
  BinaryOffsetCalculator() {
    static_assert(LhsIndexType != BroadcastIndexType::NeedCompute && RhsIndexType != BroadcastIndexType::NeedCompute);
  }

  BinaryOffsetCalculator(const int rank, const TArray<int64_t> lhs_strides, const TArray<int64_t> rhs_strides,
                         const TArray<fast_divmod> output_fdms)
      : rank_(rank), output_fdms_(output_fdms) {
    if (LhsIndexType == BroadcastIndexType::NeedCompute) {
      for (int dim = 0; dim < rank; ++dim) lhs_strides_[dim] = static_cast<CUDA_LONG>(lhs_strides[dim]);
    }
    if (RhsIndexType == BroadcastIndexType::NeedCompute) {
      for (int dim = 0; dim < rank; ++dim) rhs_strides_[dim] = static_cast<CUDA_LONG>(rhs_strides[dim]);
    }
  }

  __device__ __forceinline__ TArray<CUDA_LONG, 2> get(CUDA_LONG linear_idx) const {
    TArray<CUDA_LONG, 2> offsets;
    offsets[0] = LhsIndexType == BroadcastIndexType::NoBroadcast ? linear_idx : 0;
    offsets[1] = RhsIndexType == BroadcastIndexType::NoBroadcast ? linear_idx : 0;

    if (LhsIndexType == BroadcastIndexType::NeedCompute || RhsIndexType == BroadcastIndexType::NeedCompute) {
      CUDA_LONG q, r = linear_idx;
#pragma unroll
      for (int dim = 0; dim < output_fdms_.Capacity(); ++dim) {
        if (dim == rank_) {
          break;
        }
        output_fdms_[dim].divmod(r, q, r);
        if (LhsIndexType == BroadcastIndexType::NeedCompute) offsets[0] += lhs_strides_[dim] * q;
        if (RhsIndexType == BroadcastIndexType::NeedCompute) offsets[1] += rhs_strides_[dim] * q;
      }
    }

    return offsets;
  }

  int rank_ = 0;
  TArray<CUDA_LONG> lhs_strides_;
  TArray<CUDA_LONG> rhs_strides_;
  TArray<fast_divmod> output_fdms_;
};

template <bool IsRhsNeedCompute, bool IsMultiBatch>
struct BinaryPerChannelOffsetCalculator {
  BinaryPerChannelOffsetCalculator(const int height, const int channel) {
    fast_divmod_height_ = fast_divmod(height);
    if (IsMultiBatch) fast_divmod_channel_ = fast_divmod(channel);
  }

  __device__ __forceinline__ TArray<CUDA_LONG, 2> get(CUDA_LONG linear_idx) const {
    TArray<CUDA_LONG, 2> offsets;
    CUDA_LONG offset = fast_divmod_height_.div(linear_idx);
    if (IsMultiBatch) offset = fast_divmod_channel_.mod(offset);
    offsets[0] = IsRhsNeedCompute ? linear_idx : offset;
    offsets[1] = IsRhsNeedCompute ? offset : linear_idx;
    return offsets;
  }

  fast_divmod fast_divmod_height_;
  fast_divmod fast_divmod_channel_;
};

template <typename T, typename T1, typename T2, typename FuncT, typename OffsetCalcT>
__global__ void UnrolledBinaryElementwiseKernel(const T1* lhs_data, const T2* rhs_data, T* output_data, FuncT functor,
                                                OffsetCalcT offset_calc, CUDA_LONG N) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T1 lvalue[kElementsPerThread];
  T2 rvalue[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      TArray<CUDA_LONG, 2> offsets = offset_calc.get(id);
      lvalue[i] = lhs_data[offsets[0]];
      rvalue[i] = rhs_data[offsets[1]];
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = functor(lvalue[i], rvalue[i]);
      id += kThreadsPerBlock;
    }
  }
}

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseNoBroadcastImpl(cudaStream_t stream, const T1* lhs_data, const T2* rhs_data, T* output_data,
                                      const FuncT& func, size_t output_size) {
  if (output_size == 0) return;
  CUDA_LONG N = static_cast<CUDA_LONG>(output_size);
  int blocks_per_grid = static_cast<int>(CeilDiv(N, kThreadsPerBlock * kElementsPerThread));
  auto offset_calc = BinaryOffsetCalculator<BroadcastIndexType::NoBroadcast, BroadcastIndexType::NoBroadcast>();
  UnrolledBinaryElementwiseKernel<T, T1, T2, FuncT, decltype(offset_calc)>
      <<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(lhs_data, rhs_data, output_data, func, offset_calc, N);
}

#define LAUNCH_BINARY_ELEMENTWISE_PER_CHANNEL_KERNEL(is_rhs_need_compute, is_multi_batch)                              \
  auto offset_calc = BinaryPerChannelOffsetCalculator<is_rhs_need_compute, is_multi_batch>(args.height, args.channel); \
  UnrolledBinaryElementwiseKernel<T, T1, T2, FuncT, decltype(offset_calc)>                                             \
      <<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(lhs_data, rhs_data, output_data, func, offset_calc, N)

#define HANDLE_BINARY_ELEMENTWISE_PER_CHANNEL_BATCH(is_rhs_need_compute)      \
  if (args.is_multi_batch) {                                                  \
    LAUNCH_BINARY_ELEMENTWISE_PER_CHANNEL_KERNEL(is_rhs_need_compute, true);  \
  } else {                                                                    \
    LAUNCH_BINARY_ELEMENTWISE_PER_CHANNEL_KERNEL(is_rhs_need_compute, false); \
  }

#define HANDLE_BINARY_ELEMENTWISE_RHS_INDEX_TYPE(lhs_index_type, rhs_index_type)                                   \
  case rhs_index_type: {                                                                                           \
    auto offset_calc = BinaryOffsetCalculator<lhs_index_type, rhs_index_type>(                                     \
        static_cast<int>(args.rank), args.lhs_strides, args.rhs_strides, args.output_fdms);                        \
    UnrolledBinaryElementwiseKernel<T, T1, T2, FuncT, decltype(offset_calc)>                                       \
        <<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(lhs_data, rhs_data, output_data, func, offset_calc, N); \
  } break

#define HANDLE_BINARY_ELEMENTWISE_LHS_INDEX_TYPE(lhs_index_type, rhs_index_type_val)             \
  case lhs_index_type: {                                                                         \
    switch (rhs_index_type_val) {                                                                \
      HANDLE_BINARY_ELEMENTWISE_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::NoBroadcast); \
      HANDLE_BINARY_ELEMENTWISE_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::Scalar);      \
      HANDLE_BINARY_ELEMENTWISE_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::NeedCompute); \
    }                                                                                            \
  } break

template <typename T, typename T1, typename T2, typename FuncT>
void BinaryElementWiseImpl(cudaStream_t stream, const T1* lhs_data, const T2* rhs_data, T* output_data,
                           const BinaryElementwiseArgs& args, const FuncT& func) {
  if (args.output_size == 0) return;
  CUDA_LONG N = static_cast<CUDA_LONG>(args.output_size);
  int blocks_per_grid = static_cast<int>(CeilDiv(N, kElementsPerThread * kThreadsPerBlock));

  if (args.per_channel_type == PerChannelType::LhsNeedCompute) {
    HANDLE_BINARY_ELEMENTWISE_PER_CHANNEL_BATCH(false);
  } else if (args.per_channel_type == PerChannelType::RhsNeedCompute) {
    HANDLE_BINARY_ELEMENTWISE_PER_CHANNEL_BATCH(true);
  } else {
    switch (args.lhs_index_type) {
      HANDLE_BINARY_ELEMENTWISE_LHS_INDEX_TYPE(BroadcastIndexType::NoBroadcast, args.rhs_index_type);
      HANDLE_BINARY_ELEMENTWISE_LHS_INDEX_TYPE(BroadcastIndexType::Scalar, args.rhs_index_type);
      HANDLE_BINARY_ELEMENTWISE_LHS_INDEX_TYPE(BroadcastIndexType::NeedCompute, args.rhs_index_type);
    }
  }
}

#undef HANDLE_BINARY_ELEMENTWISE_LHS_INDEX_TYPE
#undef HANDLE_BINARY_ELEMENTWISE_RHS_INDEX_TYPE
#undef HANDLE_BINARY_ELEMENTWISE_PER_CHANNEL_BATCH
#undef LAUNCH_BINARY_ELEMENTWISE_PER_CHANNEL_KERNEL

}  // namespace cuda
}  // namespace onnxruntime
