// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/shared_inc/binary_elementwise_args.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

template <typename T>
void DispatchBiasSoftmaxForwardImpl(cudaStream_t stream, T* output_data, const T* input_data, const T* bias_data,
                                    int element_count, int batch_count, int batch_stride,
                                    int bias_broadcast_size_per_batch);

template <typename T>
Status DispatchBiasSoftMaxForwardViaDnnLibraryImpl(cudaStream_t stream, cudnnHandle_t cudaDnnHandle, int element_count,
                                                   int batch_count, int broadcast_axis, int softmax_axis,
                                                   const T* X_data, const T* B_data, T* Y_data,
                                                   const BinaryElementwiseArgs& args);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
