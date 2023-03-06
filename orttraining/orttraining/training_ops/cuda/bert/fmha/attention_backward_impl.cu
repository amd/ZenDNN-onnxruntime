// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/bert/fmha/attention_impl.h"

#include "orttraining/training_ops/cuda/bert/fmha/cutlassB.h"
#include "orttraining/training_ops/cuda/bert/fmha/kernel_backward.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status LaunchMemEfficentAttentionBwdKernel(const MemEfficientAttentionBwdParams& params) {
  const bool use_dropout = params.p > 0.0f;

  const int compute_capability = params.ComputeCapability();
  bool kernel_launched = false;
  const auto max_shmem = getMaximumSharedMemoryPerBlockKb(compute_capability) * 1024;
  const auto max_k = std::max(params.k, params.kv);

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (Kernel::kMaxK < max_k) {
      return;
    }
    // Dropout must be supported if we need it
    if (use_dropout && !Kernel::kApplyDropout) {
      return;
    }
    // Alignment
    if (params.k % Kernel::kMinimumAlignment != 0 || params.k % Kernel::kMinimumAlignment != 0 ||
        params.kv % Kernel::kMinimumAlignment != 0) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > max_shmem) {
      return;
    }

    kernel_launched = true;

    // TODO: Fuse this into a kernel?
    // This is a bottleneck for smaller sequences (M <= 128)
    // auto delta = Kernel::kKernelComputesDelta
    //                  ? at::empty({B, nH, M}, query.options().dtype(at::ScalarType::Float))
    //                  : (grad_out.to(at::kFloat) * out.to(at::kFloat)).sum(-1).transpose(-2, -1).contiguous();
    // TORCH_INTERNAL_ASSERT(delta.size(0) == B);
    // TORCH_INTERNAL_ASSERT(delta.size(1) == nH);
    // TORCH_INTERNAL_ASSERT(delta.size(2) == M);
    // TODO: support kKernelComputesDelta.
    ORT_ENFORCE(Kernel::kKernelComputesDelta, "Delta must be computed by kernel.");
    auto delta_buffer = params.kernel->GetScratchBuffer<float>(params.b * params.num_heads * params.m, params.stream);

    typename Kernel::Params p;
    p.query_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.query));
    p.key_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.key));
    p.value_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.value));
    p.logsumexp_ptr = const_cast<float*>(params.log_sum_exp_bwd);
    p.output_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.output_bwd));
    p.grad_output_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.grad_output));
    p.grad_query_ptr = reinterpret_cast<scalar_t*>(params.grad_query);
    p.grad_key_ptr = reinterpret_cast<scalar_t*>(params.grad_key);
    p.grad_value_ptr = reinterpret_cast<scalar_t*>(params.grad_value);
    p.delta_ptr = delta_buffer.get();

    p.head_dim = params.k;
    p.head_dim_value = params.kv;
    p.num_queries = params.m;
    p.num_keys = params.n;
    p.num_batches = params.b;
    p.num_heads = params.num_heads;

    // TODO: support causal.
    p.causal = false;

    p.scale = params.scale;

    p.lse_strideM = params.lse_stride;
    p.gO_strideB = params.OStrideB();
    p.gO_strideM = params.VStrideM();
    p.gO_strideH = params.VStrideH();

    p.o_strideB = params.OStrideB();
    p.o_strideH = params.VStrideH();

    p.gQ_strideB = params.QStrideB();
    p.gK_strideB = params.KStrideB();
    p.gV_strideB = params.VStrideB();
    p.gQ_strideH = params.QKStrideH();
    p.gK_strideH = params.QKStrideH();
    p.gV_strideH = params.VStrideH();
    p.gQKV_strideM_multiplier = 1;
    ORT_ENFORCE(p.gQ_strideM() == params.QKStrideM());
    ORT_ENFORCE(p.gK_strideM() == params.QKStrideM());
    ORT_ENFORCE(p.gV_strideM() == params.VStrideM());

    p.q_strideB = params.QStrideB();
    p.k_strideB = params.KStrideB();
    p.v_strideB = params.VStrideB();
    p.q_strideM = params.QKStrideM();
    p.k_strideM = params.QKStrideM();
    p.v_strideM = params.VStrideM();
    p.q_strideH = params.QKStrideH();
    p.k_strideH = params.QKStrideH();
    p.v_strideH = params.VStrideH();

    if (params.bias) {
      p.bias_ptr = const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.bias));
      p.bias_strideB = params.bias_stride_b;
      p.bias_strideH = params.bias_stride_h;
      p.bias_strideM = params.bias_stride_m;
      if (params.grad_bias) {
        p.grad_bias_ptr = reinterpret_cast<scalar_t*>(params.grad_bias);
        p.gB_strideB = params.bias_stride_b;
        p.gB_strideH = params.bias_stride_h;
        p.gB_strideM = params.bias_stride_m;
      }
    }

    if (use_dropout) {
      p.seed_offset = std::make_pair(params.seed, params.offset);
      p.dropout_prob = params.p;
    }

    int64_t size_bytes = p.workspace_size();
    IAllocatorUniquePtr<void> workspace;
    if (size_bytes > 0) {
      workspace = params.kernel->GetScratchBuffer<void>(size_bytes, params.stream);
      p.workspace = reinterpret_cast<float*>(workspace.get());
    }
    Kernel::check_supported(p);

    if (smem_bytes > 0xc000) {
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
      auto err = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      ORT_ENFORCE(err != cudaErrorInvalidValue, "This GPU does not have enough shared-memory (kernel requires ",
                  smem_bytes / 1024, " kb)");
      ORT_ENFORCE(err == cudaSuccess, "cudaFuncSetAttribute failed.");
    }

    // second syntax resulted in the error below on windows
    // error C3495: 'kernel_fn': a simple capture must be a variable
    // with automatic storage duration declared
    // in the reaching scope of the lambda
    // #ifdef _WIN32
    //     cudaFuncAttributes attr;
    //     AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
    //     TORCH_INTERNAL_ASSERT(attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability,
    //                           "Something went wrong in the build process");
    // #else
    //     auto checkBinaryArchMatches = [&]() {
    //       cudaFuncAttributes attr;
    //       AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
    //       return attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability;
    //     };
    //     TORCH_INTERNAL_ASSERT(checkBinaryArchMatches(), "Something went wrong in the build process");
    // #endif

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, params.CudaStream()>>>(p);
  };

  dispatch_cutlassB<T>(launchKernel, compute_capability);
  ORT_ENFORCE(kernel_launched, "cutlassB: no kernel found to launch!");
  return CUDA_CALL(cudaGetLastError());
}

Status MemEfficentAttentionBackward(const MemEfficientAttentionBwdParams& params) {
  if (params.is_half) {
    return LaunchMemEfficentAttentionBwdKernel<cutlass::half_t>(params);
  }

  return LaunchMemEfficentAttentionBwdKernel<float>(params);
}

}  // namespace cuda
}  // namespace onnxruntime
