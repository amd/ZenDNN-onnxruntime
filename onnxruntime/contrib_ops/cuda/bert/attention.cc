// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_attention.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Attention,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

static inline bool HasFusedFp16Kernel(int sm, int head_size, int sequence_length) {
  if (!(sm == kSM_70 || sm == kSM_75 || sm == kSM_80 || sm == kSM_86)) {
    return false;
  }

  if (head_size != 64) {
    return false;
  }

  // For sequence length 512, SM86 could fall back to SM80.
  // In our test, T4 GPU has no enough shared memory to load fmha_v2_fp16_512_64_sm75_kernel so we removed it.
  const int max_sequence_length = (sm >= kSM_80 ? 512 : 384);
  if (sequence_length > max_sequence_length) {
    return false;
  }

  return true;
}

static inline bool HasFlashAttentionKernel(int sm, int head_size) {
  if (!(sm == kSM_75 || sm == kSM_80 || sm == kSM_86)) {
    return false;
  }

  // Head size can be 8, 16, 24, ..., 128
  if (head_size > 128 || head_size % 8 != 0) {
    return false;
  }

  return true;
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info, false, false) {
  disable_fused_runner_ = sizeof(T) != 2 ||
                          ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFusedAttention, false);
  enable_flash_attention_ = sizeof(T) == 2 &&
                            ParseEnvironmentVariableWithDefault<bool>(attention::kEnableFlashAttention, false);
  enable_unpad_attention_ = ParseEnvironmentVariableWithDefault<bool>(attention::kEnableUnpadAttention, false);
  enable_dump_ = ParseEnvironmentVariableWithDefault<bool>(attention::kEnableDumpAttention, false);
}

template <typename T>
Status Attention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);
  const Tensor* extra_add_qk = context->Input<Tensor>(5);
  const Tensor* key = context->Input<Tensor>(6);
  const Tensor* value = context->Input<Tensor>(7);

  transformers::CudaTensorConsoleDumper dumper;
  if (!this->enable_dump_) {
    dumper.Disable();
  }

  auto& device_prop = GetDeviceProp();
  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  nullptr == weights ? nullptr : &(weights->Shape()),
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  extra_add_qk,
                                  key,
                                  value,
                                  &parameters,
                                  device_prop.maxThreadsPerBlock));

  int batch_size = parameters.batch_size;
  int sequence_length = parameters.sequence_length;

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{2, parameters.batch_size, parameters.num_heads,
                                    parameters.total_sequence_length, parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(1, present_shape);

  // Check whether we can use fused kernel
  int sm = device_prop.major * 10 + device_prop.minor;
  bool is_1d_mask = nullptr != mask_index && mask_index->Shape().NumDimensions() == 1;
  bool use_flash_attention = enable_flash_attention_ &&
                             nullptr != weights &&
                             (nullptr == mask_index || (is_1d_mask && enable_unpad_attention_)) &&
                             nullptr == past &&
                             nullptr == present &&
                             nullptr == extra_add_qk &&
                             !is_unidirectional_ &&
                             parameters.hidden_size == parameters.v_hidden_size &&
                             parameters.sequence_length == parameters.kv_sequence_length &&
                             HasFlashAttentionKernel(sm, parameters.head_size);

  MHARunner* fused_runner = nullptr;
  if (!use_flash_attention) {
    bool use_fused_runner = (!disable_fused_runner_ &&
                             (nullptr == mask_index || is_1d_mask) &&
                             nullptr == past &&
                             nullptr == present &&
                             nullptr == extra_add_qk &&
                             !is_unidirectional_ &&
                             parameters.hidden_size == parameters.v_hidden_size &&
                             parameters.sequence_length == parameters.kv_sequence_length &&
                             HasFusedFp16Kernel(sm, parameters.head_size, sequence_length));

    if (use_fused_runner) {
      if (nullptr == fused_fp16_runner_.get()) {
        fused_fp16_runner_.reset(new FusedMHARunnerFP16v2(num_heads_, parameters.head_size, sm));
      }

      // In case some kernel not loaded due to shared memory limit, we need to double check here.
      const int S = fused_fp16_runner_->getSFromMaxSeqLen(sequence_length);
      if (fused_fp16_runner_->isValid(S)) {
        fused_runner = fused_fp16_runner_.get();
      }
    }
  }

  constexpr size_t element_size = sizeof(T);
  size_t workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   parameters.kv_sequence_length,
                                                   parameters.total_sequence_length,
                                                   fused_runner,
                                                   use_flash_attention);
  auto work_space = GetScratchBuffer<void>(workSpaceSize);

  cublasHandle_t cublas = CublasHandle();

  typedef typename ToCudaType<T>::MappedType CudaT;

  IAllocatorUniquePtr<T> gemm_buffer;
  if (!use_flash_attention && weights != nullptr) {
    int m = batch_size * sequence_length;
    int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
    int k = parameters.input_hidden_size;
    gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n);

    CudaT one = ToCudaType<T>::FromFloat(1.0f);
    CudaT zero = ToCudaType<T>::FromFloat(0.0f);

    // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
    // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
        reinterpret_cast<const CudaT*>(input->Data<T>()), k,
        &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

    dumper.Print("gemm_buffer", gemm_buffer.get(), m, n);
  }

  typedef typename ToCudaType<T>::MappedType CudaT;
  AttentionData<CudaT> data;
  data.gemm_buffer = (nullptr == weights) ? nullptr : reinterpret_cast<const CudaT*>(gemm_buffer.get());
  data.bias = reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.query = (nullptr != weights) ? nullptr : reinterpret_cast<const CudaT*>(input->Data<T>());
  data.key = (nullptr == key) ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = (nullptr == value) ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.mask_index = (nullptr == mask_index) ? nullptr : mask_index->Data<int>();
  data.mask_index_dims = (nullptr == mask_index) ? gsl::span<const int64_t>() : mask_index->Shape().GetDims();
  data.past = (nullptr == past) ? nullptr : reinterpret_cast<const CudaT*>(past->Data<T>());
  data.extra_add_qk = (nullptr == extra_add_qk) ? nullptr : reinterpret_cast<const CudaT*>(extra_add_qk->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present = (nullptr == present) ? nullptr : reinterpret_cast<CudaT*>(present->MutableData<T>());

  auto stream = Stream();
  if (!use_flash_attention) {
    auto status = QkvToContext<CudaT>(device_prop, cublas, stream, parameters, data,
                                      reinterpret_cast<void*>(fused_runner));
    dumper.Print("output", output->MutableData<T>(), batch_size, sequence_length, parameters.v_hidden_size);
    return status;
  }

  if (nullptr == data.mask_index) {
    const size_t cumulated_seq_len_elements = ((nullptr == data.mask_index) ? batch_size : 2 * batch_size) + 1;
    auto cumulated_seq_len_buffer = GetScratchBuffer<int>(cumulated_seq_len_elements);
    int* sequence_offset = reinterpret_cast<int*>(cumulated_seq_len_buffer.get());
    LaunchTrtSequenceOffset(sequence_offset, data.mask_index, batch_size, sequence_length, stream);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    dumper.Print("cumulated_seq_len", cumulated_seq_len_buffer.get(), 1, cumulated_seq_len_elements);

    int total_token_count = batch_size * sequence_length;
    int max_token_count = sequence_length;

    typedef typename ToCudaType<T>::MappedType CudaT;

    int m = total_token_count;
    int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
    int k = parameters.input_hidden_size;
    gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n);

    CudaT one = ToCudaType<T>::FromFloat(1.0f);
    CudaT zero = ToCudaType<T>::FromFloat(0.0f);

    // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x compressed_input + 1 x bias
    // The bias part is not included here since we fuse bias and output 3 matrice into one cuda kernel.
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
        reinterpret_cast<const CudaT*>(input->Data<T>()), k,
        &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

    data.gemm_buffer = reinterpret_cast<const CudaT*>(gemm_buffer.get());
    dumper.Print("gemm_buffer", gemm_buffer.get(), m, n);

    int max_seqlen_q_ = max_token_count;
    size_t softmax_lse_bytes = get_softmax_lse_size(max_seqlen_q_, batch_size, parameters.num_heads);
    auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes);

    auto fmha_output_buffer = GetScratchBuffer<T>(static_cast<size_t>(total_token_count) * parameters.v_hidden_size);

    const size_t elements_q = static_cast<size_t>(total_token_count) * static_cast<size_t>(parameters.hidden_size);
    const size_t elements_k = static_cast<size_t>(total_token_count) * static_cast<size_t>(parameters.hidden_size);
    const size_t elements_v = static_cast<size_t>(total_token_count) * static_cast<size_t>(parameters.v_hidden_size);

    CudaT* q_data = data.workspace;
    CudaT* k_data = q_data + elements_q;
    CudaT* v_data = k_data + elements_k;
    CudaT* o_tmp_buffer = v_data + elements_v;

    const int format = 3;
    // format 3: BxSx(NH + NH + NH_v) => BxSxNxH + BxSxNxH + BxSxNxH_v
    LaunchAddBiasTranspose(stream, 3, format, device_prop.maxThreadsPerBlock,
                          batch_size, sequence_length, parameters.num_heads, parameters.head_size,
                          data.gemm_buffer, data.bias, data.workspace,
                          true, -1);

    dumper.Print("q", reinterpret_cast<T*>(q_data), total_token_count, parameters.hidden_size);
    dumper.Print("k", reinterpret_cast<T*>(k_data), total_token_count, parameters.hidden_size);
    dumper.Print("v", reinterpret_cast<T*>(v_data), total_token_count, parameters.v_hidden_size);

    const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(parameters.head_size));

    fmha_forward(device_prop,
                stream,
                reinterpret_cast<void*>(q_data),
                reinterpret_cast<void*>(k_data),
                reinterpret_cast<void*>(v_data),
                reinterpret_cast<void*>(output->MutableData<T>()),
                cumulated_seq_len_buffer.get(),
                cumulated_seq_len_buffer.get(),
                reinterpret_cast<void*>(softmax_lse_buffer.get()),
                reinterpret_cast<void*>(o_tmp_buffer),
                batch_size,
                parameters.num_heads,
                parameters.head_size,
                parameters.v_head_size,
                total_token_count,  // total_q
                max_token_count,    // max_seqlen_q_,
                max_token_count,    // max_seqlen_k_,
                rsqrt_head_size,    // softmax_scale,
                false,              // is_causal,
                0                   // num_splits
    );

    dumper.Print("fmha_output", output->MutableData<T>(), total_token_count, parameters.v_hidden_size);
    return Status::OK();
  }

  // Remove padding for flash attention
  ORT_ENFORCE(this->enable_unpad_attention_);
  //if (this->enable_unpad_attention_)
  {
    auto token_count_buffer = GetScratchBuffer<int>(2);
    auto token_offset_buffer = GetScratchBuffer<int>(batch_size * sequence_length);
    auto cumulated_seq_len_buffer = GetScratchBuffer<int>(batch_size + 1);

    LaunchGetTokenOffset(token_count_buffer.get(),
                         token_offset_buffer.get(),
                         cumulated_seq_len_buffer.get(),
                         data.mask_index,
                         batch_size,
                         sequence_length,
                         stream);

    dumper.Print("token_count", token_count_buffer.get(), 1, 2);
    dumper.Print("token_offset", token_offset_buffer.get(), batch_size, sequence_length);
    dumper.Print("cumulated_seq_len", cumulated_seq_len_buffer.get(), 1, batch_size + 1);

    // Copy token_count to CPU
    auto pinned_buffer = AllocateBufferOnCPUPinned<int>(2);
    int* token_count_pinned = pinned_buffer.get();
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(token_count_pinned, token_count_buffer.get(), sizeof(int) * 2,
                                         cudaMemcpyDeviceToHost, stream));
    // Wait until token_count is copied to host.
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    int total_token_count = token_count_pinned[0];
    int max_token_count = token_count_pinned[1];

    auto compressed_input_buffer = GetScratchBuffer<T>(total_token_count * parameters.input_hidden_size);

    typedef typename ToCudaType<T>::MappedType CudaT;
    LaunchRemovePadding<CudaT>(
        reinterpret_cast<CudaT*>(compressed_input_buffer.get()),
        reinterpret_cast<const CudaT*>(input->Data<T>()),
        token_offset_buffer.get(),
        total_token_count,
        parameters.input_hidden_size,
        stream);

    dumper.Print("compressed_input", compressed_input_buffer.get(), total_token_count, parameters.input_hidden_size);

    int m = total_token_count;
    int n = (parameters.hidden_size + parameters.hidden_size + parameters.v_hidden_size);
    int k = parameters.input_hidden_size;
    gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n);

    CudaT one = ToCudaType<T>::FromFloat(1.0f);
    CudaT zero = ToCudaType<T>::FromFloat(0.0f);

    // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x compressed_input + 1 x bias
    // The bias part is not included here since we fuse bias and output 3 matrice into one cuda kernel.
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
        reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
        reinterpret_cast<const CudaT*>(compressed_input_buffer.get()), k,
        &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

    data.gemm_buffer = reinterpret_cast<const CudaT*>(gemm_buffer.get());
    dumper.Print("gemm_buffer", gemm_buffer.get(), m, n);

    int max_seqlen_q_ = max_token_count;
    size_t softmax_lse_bytes = get_softmax_lse_size(max_seqlen_q_, batch_size, parameters.num_heads);
    auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes);

    auto fmha_output_buffer = GetScratchBuffer<T>(static_cast<size_t>(total_token_count) * parameters.v_hidden_size);

    const size_t elements_q = static_cast<size_t>(total_token_count) * static_cast<size_t>(parameters.hidden_size);
    const size_t elements_k = static_cast<size_t>(total_token_count) * static_cast<size_t>(parameters.hidden_size);
    const size_t elements_v = static_cast<size_t>(total_token_count) * static_cast<size_t>(parameters.v_hidden_size);

    CudaT* q_data = data.workspace;
    CudaT* k_data = q_data + elements_q;
    CudaT* v_data = k_data + elements_k;
    CudaT* o_tmp_buffer = v_data + elements_v;

    const int format = 3;
    // format 3: BxSx(NH + NH + NH_v) => BxSxNxH + BxSxNxH + BxSxNxH_v
    LaunchAddBiasTranspose(stream, 3, format, device_prop.maxThreadsPerBlock,
                           1, total_token_count, parameters.num_heads, parameters.head_size,
                           data.gemm_buffer, data.bias, data.workspace,
                           true, -1);

    dumper.Print("q", reinterpret_cast<T*>(q_data), total_token_count, parameters.hidden_size);
    dumper.Print("k", reinterpret_cast<T*>(k_data), total_token_count, parameters.hidden_size);
    dumper.Print("v", reinterpret_cast<T*>(k_data), total_token_count, parameters.v_hidden_size);

    const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(parameters.head_size));

    fmha_forward(device_prop,
                 stream,
                 reinterpret_cast<void*>(q_data),
                 reinterpret_cast<void*>(k_data),
                 reinterpret_cast<void*>(v_data),
                 reinterpret_cast<void*>(fmha_output_buffer.get()),
                 cumulated_seq_len_buffer.get(),
                 cumulated_seq_len_buffer.get(),
                 reinterpret_cast<void*>(softmax_lse_buffer.get()),
                 reinterpret_cast<void*>(o_tmp_buffer),
                 batch_size,
                 parameters.num_heads,
                 parameters.head_size,
                 parameters.v_head_size,
                 total_token_count,  // total_q
                 max_token_count,    // max_seqlen_q_,
                 max_token_count,    // max_seqlen_k_,
                 rsqrt_head_size,    // softmax_scale,
                 false,              // is_causal,
                 0                   // num_splits
    );

    dumper.Print("fmha_output", fmha_output_buffer.get(), total_token_count, parameters.v_hidden_size);

    // Restore padding
    LaunchRestorePadding<CudaT>(
        reinterpret_cast<CudaT*>(output->MutableData<T>()),
        reinterpret_cast<const CudaT*>(fmha_output_buffer.get()),
        token_offset_buffer.get(),
        total_token_count,
        parameters.v_hidden_size,
        batch_size,
        sequence_length,
        stream);

    dumper.Print("output", output->MutableData<T>(), batch_size, sequence_length, parameters.v_hidden_size);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
