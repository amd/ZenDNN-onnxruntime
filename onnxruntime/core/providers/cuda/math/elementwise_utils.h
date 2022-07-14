// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <initializer_list>
#include <functional>

#include "core/framework/tensor_shape.h"

namespace onnxruntime {

namespace common {
class Status;
}  // namespace common

namespace cuda {

/**
 * @brief Coalesce contiguous dimensions in the tensors. Operates inplace on the function arguments.
 *
 * @param tensors_strides Strides of tensors.
 * @param shape  Output tensor shape.
 */
void CoalesceDimensions(std::initializer_list<std::reference_wrapper<TensorShapeVector>>&& tensors_strides,
                        TensorShapeVector& shape);

/**
 * @brief Compute elementwise output shape from input shapes.
 *
 * @param node_name The node name for logging.
 * @param shapes The input shapes.
 * @param output_shape The output shape.
 * @return onnxruntime::common::Status It's Status::OK() if success.
 */
onnxruntime::common::Status ComputeOutputShape(
    const std::string& node_name, std::initializer_list<std::reference_wrapper<const TensorShape>>&& shapes,
    TensorShape& output_shape);

}  // namespace cuda
}  // namespace onnxruntime
