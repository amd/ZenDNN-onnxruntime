// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/copy_impl.h"

#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, typename OffsetCalcT>
__global__ void StridedCopyKernel(const T* src_data, T* dst_data, OffsetCalcT offset_calc, CUDA_LONG N) {
  CUDA_LONG start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T src_value[kElementsPerThread];
  CUDA_LONG dst_offset[kElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      TArray<int32_t, 2> offsets = offset_calc.get(id);
      src_value[i] = src_data[offsets[0]];
      dst_offset[i] = offsets[1];
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      dst_data[dst_offset[i]] = src_value[i];
      id += kThreadsPerBlock;
    }
  }
}

#define LAUNCH_STRIDED_COPY_PER_CHANNEL_KERNEL(is_rhs_need_compute, is_multi_batch)                                    \
  auto offset_calc = BinaryPerChannelOffsetCalculator<is_rhs_need_compute, is_multi_batch>(args.height, args.channel); \
  StridedCopyKernel<T, decltype(offset_calc)>                                                                          \
      <<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(src_data, dst_data, offset_calc, N)

#define HANDLE_STRIDED_COPY_PER_CHANNEL_BATCH(is_rhs_need_compute)      \
  if (args.is_multi_batch) {                                            \
    LAUNCH_STRIDED_COPY_PER_CHANNEL_KERNEL(is_rhs_need_compute, true);  \
  } else {                                                              \
    LAUNCH_STRIDED_COPY_PER_CHANNEL_KERNEL(is_rhs_need_compute, false); \
  }

#define HANDLE_STRIDED_COPY_RHS_INDEX_TYPE(lhs_index_type, rhs_index_type)                      \
  case rhs_index_type: {                                                                        \
    auto offset_calc = BinaryOffsetCalculator<lhs_index_type, rhs_index_type>(                  \
        static_cast<int>(args.rank), args.lhs_strides, args.rhs_strides, args.output_fdms);     \
    StridedCopyKernel<T, decltype(offset_calc)>                                                 \
        <<<blocks_per_grid, kThreadsPerBlock, 0, stream>>>(src_data, dst_data, offset_calc, N); \
  } break

#define HANDLE_STRIDED_COPY_LHS_INDEX_TYPE(lhs_index_type, rhs_index_type_val)             \
  case lhs_index_type: {                                                                   \
    switch (rhs_index_type_val) {                                                          \
      HANDLE_STRIDED_COPY_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::NoBroadcast); \
      HANDLE_STRIDED_COPY_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::Scalar);      \
      HANDLE_STRIDED_COPY_RHS_INDEX_TYPE(lhs_index_type, BroadcastIndexType::NeedCompute); \
    }                                                                                      \
  } break

template <typename T>
void StridedCopyImpl(cudaStream_t stream, const T* src_data, T* dst_data, const BinaryElementwiseArgs& args) {
  if (args.output_size == 0) return;
  CUDA_LONG N = static_cast<CUDA_LONG>(args.output_size);
  int blocks_per_grid = static_cast<int>(CeilDiv(N, kElementsPerThread * kThreadsPerBlock));
  if (args.per_channel_type == PerChannelType::LhsNeedCompute) {
    HANDLE_STRIDED_COPY_PER_CHANNEL_BATCH(false);
  } else if (args.per_channel_type == PerChannelType::RhsNeedCompute) {
    HANDLE_STRIDED_COPY_PER_CHANNEL_BATCH(true);
  } else {
    switch (args.lhs_index_type) {
      HANDLE_STRIDED_COPY_LHS_INDEX_TYPE(BroadcastIndexType::NoBroadcast, args.rhs_index_type);
      HANDLE_STRIDED_COPY_LHS_INDEX_TYPE(BroadcastIndexType::Scalar, args.rhs_index_type);
      HANDLE_STRIDED_COPY_LHS_INDEX_TYPE(BroadcastIndexType::NeedCompute, args.rhs_index_type);
    }
  }
}

#undef HANDLE_STRIDED_COPY_LHS_INDEX_TYPE
#undef HANDLE_STRIDED_COPY_RHS_INDEX_TYPE
#undef HANDLE_STRIDED_COPY_PER_CHANNEL_BATCH
#undef LAUNCH_STRIDED_COPY_PER_CHANNEL_KERNEL

#define SPECIALIZED_STRIDED_COPY_IMPL(T)                                                \
  template void StridedCopyImpl<T>(cudaStream_t stream, const T* src_data, T* dst_data, \
                                   const BinaryElementwiseArgs& args);

SPECIALIZED_STRIDED_COPY_IMPL(int8_t)
SPECIALIZED_STRIDED_COPY_IMPL(int16_t)
SPECIALIZED_STRIDED_COPY_IMPL(int32_t)
SPECIALIZED_STRIDED_COPY_IMPL(int64_t)

#undef SPECIALIZED_STRIDED_COPY_IMPL

}  // namespace cuda
}  // namespace onnxruntime
