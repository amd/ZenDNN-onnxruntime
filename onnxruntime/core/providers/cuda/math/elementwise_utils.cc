//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/elementwise_utils.h"

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"

namespace onnxruntime {
namespace cuda {

void CoalesceDimensions(std::initializer_list<std::reference_wrapper<TensorShapeVector>>&& tensors_strides,
                        TensorShapeVector& shape) {
  const std::size_t dims = shape.size();
  if (dims <= 1) return;

  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[src] * stride[src] == shape[dst].
  auto CanCoalesce = [&](int dst, int src) {
    auto shape_dst = shape[dst];
    auto shape_src = shape[src];
    if (shape_dst == 1 || shape_src == 1) {
      return true;
    }
    for (const auto& r_tensor_strides : tensors_strides) {
      auto& tensor_strides = r_tensor_strides.get();
      if (shape_src * tensor_strides[src] != tensor_strides[dst]) {
        return false;
      }
    }
    return true;
  };

  // replace each operands stride at dst with its stride at src
  auto ReplaceStride = [&](int dst, int src) {
    for (const auto& r_tensor_strides : tensors_strides) {
      auto& tensor_strides = r_tensor_strides.get();
      tensor_strides[dst] = tensor_strides[src];
    }
  };

  // the current dimension is the one we are attempting to "coalesce onto"
  std::size_t current_dim = 0;
  for (std::size_t dim = 1; dim < dims; ++dim) {
    // check if dim can be coalesced with current_dim
    if (CanCoalesce(current_dim, dim)) {
      if (shape[dim] != 1) {
        ReplaceStride(current_dim, dim);
      }
      shape[current_dim] *= shape[dim];
    } else {
      current_dim++;
      if (current_dim != dim) {
        // we have coalesced at least one value before this: bump forward the values into the correct place
        ReplaceStride(current_dim, dim);
        shape[current_dim] = shape[dim];
      }
    }
  }

  shape.resize(current_dim + 1);
  for (const auto& r_tensor_strides : tensors_strides) {
    auto& tensor_strides = r_tensor_strides.get();
    tensor_strides.resize(current_dim + 1);
  }
}

Status ComputeOutputShape(const std::string& node_name,
                          std::initializer_list<std::reference_wrapper<const TensorShape>>&& shapes,
                          TensorShape& output_shape) {
  InlinedVector<size_t> ranks;
  for (const auto& shape : shapes) {
    ranks.emplace_back(shape.get().NumDimensions());
  }

  size_t out_rank = *std::max_element(ranks.cbegin(), ranks.cend());
  TensorShapeVector output_dims(out_rank, 0);

  for (size_t i = 0; i < out_rank; ++i) {
    TensorShapeVector dims;
    size_t j = 0;
    for (const auto& shape : shapes) {
      int64_t dim = 1;
      if (i < ranks[j]) dim = shape.get()[ranks[j] - 1 - i];
      dims.emplace_back(dim);
      j++;
    }
    int64_t max = *std::max_element(dims.cbegin(), dims.cend());
    int64_t min = *std::min_element(dims.cbegin(), dims.cend());
    int64_t out_dim = (min == 0 ? min : max);  // special case a dim value of 0.
    for (j = 0; j < dims.size(); ++j) {
      if (dims[j] != out_dim && dims[j] != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": input ", j, " cannot broadcast on dim ",
                               ranks[j] - 1 - i, " from ", dims[j], " to ", out_dim);
      }
    }
    output_dims[out_rank - 1 - i] = out_dim;
  }
  output_shape = TensorShape(output_dims);
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
