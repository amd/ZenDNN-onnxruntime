// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/bert/fmha/attention_impl.h"

#include "orttraining/training_ops/cuda/bert/fmha/cutlassF.h"
#include "orttraining/training_ops/cuda/bert/fmha/kernel_forward.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status LaunchMemEfficentAttentionFwdKernel(const MemEfficientAttentionFwdParams& params) {
  const bool use_dropout = params.p > 0.0f;
  std::pair<uint64_t, uint64_t> seed_offset;
  if (use_dropout) {
    seed_offset = std::make_pair(params.seed, params.offset);
  }

  const int compute_capability = params.ComputeCapability();
  bool kernel_launched = false;
  const auto max_shmem = getMaximumSharedMemoryPerBlockKb(compute_capability) * 1024;

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (!Kernel::kSupportsDropout && use_dropout) {
      return;
    }
    if (!Kernel::kSupportsBias && params.bias) {
      return;
    }
    if (Kernel::kSingleValueIteration && Kernel::kKeysPerBlock < params.kv) {
      return;
    }
    // Alignment
    if (params.k % Kernel::kAlignmentQ != 0 || params.k % Kernel::kAlignmentK != 0 ||
        params.kv % Kernel::kAlignmentV != 0) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > max_shmem) {
      return;
    }

    kernel_launched = true;

    typename Kernel::Params p;
    p.query_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.query));
    p.key_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.key));
    p.value_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.value));
    p.logsumexp_ptr = params.log_sum_exp_fwd;
    // TODO: add kNeedsOutputAccumulatorBuffer support
    ORT_ENFORCE(!Kernel::kNeedsOutputAccumulatorBuffer, "kNeedsOutputAccumulatorBuffer not supported for now.");
    p.output_accum_ptr = nullptr;
    p.output_ptr = reinterpret_cast<scalar_t*>(params.output_fwd);

    p.num_heads = params.num_heads;
    p.head_dim = params.k;
    p.head_dim_value = params.kv;
    p.num_queries = params.m;
    p.num_keys = params.n;
    p.num_batches = params.b;

    // TODO: support causal attention.
    p.causal = false;
    p.causal_diagonal_ptr = nullptr;

    p.scale = params.scale;

    p.q_strideB = params.QStrideB();
    p.k_strideB = params.KStrideB();
    p.v_strideB = params.VStrideB();
    p.q_strideM = params.QKStrideM();
    p.k_strideM = params.QKStrideM();
    p.v_strideM = params.VStrideM();
    p.q_strideH = params.QKStrideH();
    p.k_strideH = params.QKStrideH();
    p.v_strideH = params.VStrideH();
    p.o_strideM = params.VStrideM();

    if (params.bias) {
      p.attn_bias_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.bias));
      p.bias_strideB = params.bias_stride_b;
      p.bias_strideH = params.bias_stride_h;
      p.bias_strideM = params.bias_stride_m;
    }

    p.use_dropout = use_dropout;
    if (p.use_dropout) {
      p.seed_offset = seed_offset;
      p.dropout_prob = params.p;
    }

    if (smem_bytes > 0xc000) {
      auto err = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      ORT_ENFORCE(err != cudaErrorInvalidValue, "This GPU does not have enough shared-memory (kernel requires ",
                  smem_bytes / 1024, " kb)");
      ORT_ENFORCE(err == cudaSuccess, "cudaFuncSetAttribute failed.");
    }

    Kernel::check_supported(p);
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, params.CudaStream()>>>(p);
  };

  // Dispatch to the right kernel
  dispatch_cutlassF<T>(launchKernel, compute_capability);
  ORT_ENFORCE(kernel_launched, "cutlassF: no kernel found to launch!");
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  // uint64_t -> int64_t bitwise casting as PyTorch don't support uint64_t
  // so just fake it as a int64_t
  // if (use_dropout) {
  //   std::memcpy(&seed, &rng_engine_inputs.seed_, sizeof(seed));
  //   std::memcpy(&offset, &rng_engine_inputs.offset_.val, sizeof(offset));
  // }
  return Status::OK();
}

Status MemEfficentAttentionForward(const MemEfficientAttentionFwdParams& params) {
  if (params.is_half) {
    return LaunchMemEfficentAttentionFwdKernel<cutlass::half_t>(params);
  }

  return LaunchMemEfficentAttentionFwdKernel<float>(params);
}

}  // namespace cuda
}  // namespace onnxruntime
