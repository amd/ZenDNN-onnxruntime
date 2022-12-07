/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include <cuda_fp16.h>
#include "contrib_ops/cuda/bert/flash_attention/fmha.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_attention.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {
namespace fmha {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
static inline void set_alpha_fp16(uint32_t& alpha, float norm) {
  half x = __float2half_rn(norm);
  uint16_t h = reinterpret_cast<const uint16_t&>(x);
  ushort2 h2 = {h, h};
  alpha = reinterpret_cast<const uint32_t&>(h2);
}
#pragma GCC diagnostic pop

void set_params_fprop(FMHA_fprop_params& params,
                      const size_t batch_size,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t num_heads,
                      const size_t head_size,
                      const size_t v_head_size,
                      void* q,
                      void* k,
                      void* v,
                      void* out,
                      void* cu_seqlens_q_d,
                      void* cu_seqlens_k_d,
                      void* o_tmp_d,
                      void* softmax_lse_d,
                      float softmax_scale,
                      bool is_causal,
                      int num_splits  // How many SMs per attention matrix.
) {
  memset(&params, 0, sizeof(params));

  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = out;
  params.o_tmp_ptr = o_tmp_d;

  params.q_row_stride_in_elts = num_heads * head_size;    // q.stride(0)
  params.k_row_stride_in_elts = num_heads * head_size;    // k.stride(0)
  params.v_row_stride_in_elts = num_heads * v_head_size;  // v.stride(0)
  params.o_row_stride_in_elts = num_heads * v_head_size;  // o.stride(0)
  params.o_tmp_row_stride_in_elts = num_heads * v_head_size;

  params.q_head_stride_in_elts = head_size;    // q.stride(1)
  params.k_head_stride_in_elts = head_size;    // k.stride(1)
  params.v_head_stride_in_elts = v_head_size;  // v.stride(1)
  params.o_head_stride_in_elts = v_head_size;  // o.stride(1)
  params.o_tmp_head_stride_in_elts = v_head_size;

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);

  params.s_ptr = nullptr;                                            // softmax
  params.s_stride_in_bytes = batch_size * num_heads * seqlen_k * 2;  // 2 is bytes of float16
  params.softmax_lse_ptr = softmax_lse_d;                            // softmax sum

  params.b = batch_size;
  params.h = num_heads;
  params.d = head_size;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;

  const float scale_bmm1 = softmax_scale;
  params.scale_bmm1f = scale_bmm1;
  set_alpha_fp16(params.scale_bmm1, scale_bmm1);
  params.is_causal = is_causal;
  params.num_splits = num_splits;
}

int get_max_seqlen_k(int max_seqlen_k_, int head_size, bool& loop) {
  int blocksize_c = head_size > 64 ? 128 : 256;

  // Need to round max_seqlen_k to multiples of blocksize_c
  int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
  if (max_seqlen_k <= 128) {
    max_seqlen_k = 128;
  } else if (max_seqlen_k <= 256) {
    max_seqlen_k = 256;
  }

  loop = max_seqlen_k > blocksize_c;

  return max_seqlen_k;
}

int get_max_seqlen_q(int max_seqlen_q_) {
  return ((max_seqlen_q_ + 16 - 1) / 16) * 16;
}

size_t get_softmax_lse_size(int max_seqlen_q_, int batch_size, int num_heads) {
  int max_seqlen_q = get_max_seqlen_q(max_seqlen_q_);
  size_t bytes = sizeof(float) * batch_size * num_heads * max_seqlen_q;

  return bytes;
}

size_t get_o_tmp_size(int max_seqlen_k_, int total_q, int num_heads, int head_size, int v_head_size) {
  bool loop = false;
  get_max_seqlen_k(max_seqlen_k_, head_size, loop);
  return loop ? (sizeof(float) * total_q * num_heads * v_head_size) : 0;
}

Status run_fmha_fwd(Launch_params<FMHA_fprop_params>& launch_params) {
  if (launch_params.params.d <= 32) {
    return run_fmha_fwd_hdim32(launch_params);
  } else if (launch_params.params.d <= 64) {
    return run_fmha_fwd_hdim64(launch_params);
  } else if (launch_params.params.d <= 128) {
    return run_fmha_fwd_hdim128(launch_params);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "headsize > 128 is not supported by flash attention");
  }
}

Status fmha_forward(const cudaDeviceProp& dprops,
                    cudaStream_t stream,
                    void* q,                   // half (total_q, num_heads, head_size)
                    void* k,                   // half (total_k, num_heads, head_size)
                    void* v,                   // half (total_k, num_heads, v_head_size)
                    void* out,                 // half (total_q, num_heads, v_head_size)
                    int32_t* cu_seqlens_q,     // int (batch_size + 1)
                    int32_t* cu_seqlens_k,     // int (batch_size + 1)
                    void* softmax_lse_buffer,  // float (batch_size, num_heads, max_seqlen_q)
                    void* o_tmp_buffer,        // NULL or float (total_q, num_heads, v_head_size)
                    const int batch_size,
                    const int num_heads,
                    const int head_size,
                    const int v_head_size,
                    const int total_q,
                    const int max_seqlen_q_,
                    const int max_seqlen_k_,
                    const float softmax_scale,
                    const bool is_causal,
                    const int num_splits) {
  ORT_UNUSED_PARAMETER(total_q);

  bool is_sm75 = dprops.major == 7 && dprops.minor == 5;
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  ORT_ENFORCE(is_sm8x || is_sm75);

  constexpr bool return_softmax = false;
  Launch_params<FMHA_fprop_params> launch_params(&dprops, stream, return_softmax);

  ORT_ENFORCE(batch_size > 0);
  ORT_ENFORCE((head_size % 8 == 0) && (head_size <= 128));

  bool loop = false;
  int max_seqlen_k = get_max_seqlen_k(max_seqlen_k_, head_size, loop);
  int max_seqlen_q = get_max_seqlen_q(max_seqlen_q_);

  set_params_fprop(launch_params.params,
                   batch_size,
                   max_seqlen_q,
                   max_seqlen_k,
                   num_heads,
                   head_size,
                   v_head_size,
                   q, k, v, out,
                   cu_seqlens_q,
                   cu_seqlens_k,
                   loop ? o_tmp_buffer : nullptr,
                   softmax_lse_buffer,
                   softmax_scale,
                   is_causal,
                   num_splits);

  return run_fmha_fwd(launch_params);
}

}  // namespace fmha
}  // namespace cuda
}  // namespace onnxruntime
