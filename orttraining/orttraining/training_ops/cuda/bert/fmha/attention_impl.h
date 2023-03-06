// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

struct MemEfficientAttentionParams {
  // Float or Float16.
  bool is_half;

  // Inputs.
  const void* query;
  const void* key;
  const void* value;
  const void* bias;

  float p;
  uint64_t seed;
  uint64_t offset;
  float scale;

  // Attributes.
  int64_t b;
  int64_t m;
  int64_t n;
  int64_t num_heads;
  int64_t k;
  int64_t kv;
  int64_t bias_stride_b;
  int64_t bias_stride_h;
  int64_t bias_stride_m;

  const CudaKernel* kernel;
  Stream* stream;

  int ComputeCapability() const {
    const cudaDeviceProp& prop = kernel->GetDeviceProp();
    return prop.major * 10 + prop.minor;
  }

  cudaStream_t CudaStream() const { return static_cast<cudaStream_t>(stream->GetHandle()); }

  int64_t QStrideB() const { return m * num_heads * k; }
  int64_t QKStrideM() const { return num_heads * k; }
  int64_t KStrideB() const { return n * num_heads * k; }
  int64_t VStrideB() const { return n * num_heads * kv; }
  int64_t VStrideM() const { return num_heads * kv; }
  int64_t QKStrideH() const { return k; }
  int64_t VStrideH() const { return kv; }
};

struct MemEfficientAttentionFwdParams : public MemEfficientAttentionParams {
  // Fwd output.
  void* output_fwd;
  float* log_sum_exp_fwd;
};

struct MemEfficientAttentionBwdParams : public MemEfficientAttentionParams {
  // Bwd extra input.
  const void* grad_output;
  const void* output_bwd;
  const float* log_sum_exp_bwd;

  // Bwd extra attribute.
  int64_t lse_stride;

  // Bwd output.
  void* grad_query;
  void* grad_key;
  void* grad_value;
  void* grad_bias;

  int64_t OStrideB() const { return m * num_heads * kv; }
};

Status MemEfficentAttentionForward(const MemEfficientAttentionFwdParams& params);

Status MemEfficentAttentionBackward(const MemEfficientAttentionBwdParams& params);

}  // namespace cuda
}  // namespace onnxruntime
