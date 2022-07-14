// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/providers/cuda/shared_inc/binary_elementwise_args.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename VariadicElementwiseOpTag>
void Impl_General(cudaStream_t stream, const T* lhs_data, const T* rhs_data, T* output_data,
                  const BinaryElementwiseArgs& args);

constexpr int32_t k_max_input_batch_size = 8;

template <typename T>
using InputBatchArray = TArray<const T*, k_max_input_batch_size>;

template <typename T, typename VariadicElementwiseOpTag>
void Impl_NoBroadcastInputBatch(
    cudaStream_t stream,
    InputBatchArray<T> input_data_batch,
    T* output_data,
    size_t count);

}  // namespace cuda
}  // namespace onnxruntime
