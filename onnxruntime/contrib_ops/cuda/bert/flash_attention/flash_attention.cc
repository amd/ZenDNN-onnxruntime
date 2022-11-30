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
#include "contrib_ops/cuda/bert/flash_attention/fmha.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_attention.h"

// #define CHECK_SHAPE(x, ...) ORT_ENFORCE(x->Shape().AsShapeVector() == onnxruntime::TensorShapeVector({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

// int64_t get_stride(const onnxruntime::TensorShape& input_shape, size_t axis){
//     size_t rank = input_shape.NumDimensions();
//     auto input_dimensions = input_shape.GetDims();

//     ORT_ENFORCE(axis < rank);

//     int64_t stride = 1;
//     for(size_t i = axis + 1; i < rank; i++){
//         stride *= input_dimensions[i];
//     }

//     return stride;
// }

void set_params_fprop(FMHA_fprop_params& params,
                      const size_t batch_size,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t num_heads,
                      const size_t head_size,
                      // onnxruntime::Tensor* q,
                      // onnxruntime::Tensor* k,
                      // onnxruntime::Tensor* v,
                      // onnxruntime::Tensor* out,
                      void* q,
                      void* k,
                      void* v,
                      void* out,
                      void* cu_seqlens_q_d,
                      void* cu_seqlens_k_d,
                      void* o_tmp_d,
                      void* s_d,
                      void* softmax_lse_d,
                      float softmax_scale,
                      bool is_causal,
                      int num_splits  // How many SMs per attention matrix.
) {
  // Data_type acc_type = DATA_TYPE_FP32;
  Data_type data_type = DATA_TYPE_FP16;

  memset(&params, 0, sizeof(params));
  params.is_bf16 = false;

  // params.q_ptr = q->MutableDataRaw();
  // params.k_ptr = k->MutableDataRaw();
  // params.v_ptr = v->MutableDataRaw();
  // params.q_row_stride_in_elts = get_stride(q->Shape(), 0);
  // params.k_row_stride_in_elts = get_stride(k->Shape(), 0);
  // params.v_row_stride_in_elts = get_stride(v->Shape(), 0);
  // params.q_head_stride_in_elts = get_stride(q->Shape(), 1);
  // params.k_head_stride_in_elts = get_stride(k->Shape(), 1);
  // params.v_head_stride_in_elts = get_stride(v->Shape(), 1);
  // params.o_ptr = out->MutableDataRaw();
  // params.o_row_stride_in_elts = get_stride(out->Shape(), 0);
  // params.o_head_stride_in_elts = get_stride(out->Shape(), 1);
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = out;
  params.o_tmp_ptr = o_tmp_d;

  params.q_row_stride_in_elts = num_heads * head_size;
  params.k_row_stride_in_elts = num_heads * head_size;
  params.v_row_stride_in_elts = num_heads * head_size;
  params.o_row_stride_in_elts = num_heads * head_size;
  params.o_tmp_row_stride_in_elts = num_heads * head_size;

  params.q_head_stride_in_elts = head_size;
  params.k_head_stride_in_elts = head_size;
  params.v_head_stride_in_elts = head_size;
  params.o_head_stride_in_elts = head_size;
  params.o_tmp_head_stride_in_elts = head_size;

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  // S = softmax(P)
  params.s_ptr = s_d;
  params.s_stride_in_bytes = get_size_in_bytes(batch_size * num_heads * seqlen_k, data_type);
  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;
  // Set the dimensions.
  params.b = batch_size;
  params.h = num_heads;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = head_size;
  // Set the different scale values.
  // const float scale_bmm1 = 1.f / sqrtf(d);
  const float scale_bmm1 = softmax_scale;
  params.scale_bmm1f = scale_bmm1;
  set_alpha(params.scale_bmm1, scale_bmm1, data_type);
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


size_t get_flash_attention_workspace_size(int max_seqlen_q_, int max_seqlen_k_, int batch_size, int total_q, int num_heads, int head_size) {
  bool loop = false;
  get_max_seqlen_k(max_seqlen_k_, head_size, loop);
  int max_seqlen_q = get_max_seqlen_q(max_seqlen_q_);

  size_t bytes = 0;
  if (loop) {
    bytes += sizeof(float) * total_q * num_heads * head_size;
    bytes += sizeof(float) * batch_size * num_heads * max_seqlen_q;
  }

  return bytes;
}

void run_fmha_fwd(Launch_params<FMHA_fprop_params> &launch_params) {
    if (launch_params.params.d <= 32) {
        run_fmha_fwd_hdim32(launch_params);
    } else if (launch_params.params.d <= 64) {
        run_fmha_fwd_hdim64(launch_params);
    } else if (launch_params.params.d <= 128) {
        run_fmha_fwd_hdim128(launch_params);
    }
}

void fmha_forward(const cudaDeviceProp& dprops,
                  cudaStream_t stream,
                  void* q,    // shape: (total_q, num_heads, head_size)
                  void* k,    // shape: (total_k, num_heads, head_size)
                  void* v,    // shape: (total_k, num_heads, head_size)
                  void* out,  // shape: (total_q, num_heads, head_size)
                  int32_t* cu_seqlens_q,     // shape: (batch_size + 1)
                  int32_t* cu_seqlens_k,     // shape: (batch_size + 1)
                  void* workspace,
                  const int batch_size,
                  const int num_heads,
                  const int head_size,
                  const int total_q,
                  const int max_seqlen_q_,
                  const int max_seqlen_k_,
                  const float softmax_scale,
                  const bool zero_tensors,
                  const bool is_causal,
                  const int num_splits) {
  bool is_sm75 = dprops.major == 7 && dprops.minor == 5;
  bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
  ORT_ENFORCE(is_sm8x || is_sm75);

  constexpr bool is_dropout = false;
  constexpr bool return_softmax = false;
  Launch_params<FMHA_fprop_params> launch_params(&dprops, stream, is_dropout, return_softmax);
  // ORT_ENFORCE(q->IsDataType<onnxruntime::MLFloat16>());
  // ORT_ENFORCE(k->IsDataType<onnxruntime::MLFloat16>());
  // ORT_ENFORCE(v->IsDataType<onnxruntime::MLFloat16>());
  // ORT_ENFORCE(out->IsDataType<onnxruntime::MLFloat16>());
  // ORT_ENFORCE(cu_seqlens_q->DataType() == onnxruntime::DataTypeImpl::GetType<int32_t>());
  // ORT_ENFORCE(cu_seqlens_k->DataType() == onnxruntime::DataTypeImpl::GetType<int32_t>());

  // const auto dims = q->Shape().GetDims();
  // ORT_ENFORCE(dims.size() == 3);
  // const int total_q = dims[0];
  // const int num_heads = dims[1];
  // const int head_size = dims[2];
  // const int total_k = k->Shape().GetDims()[0];

  // ORT_ENFORCE(cu_seqlens_q->Shape().NumDimensions() == 1);
  // const int batch_size = cu_seqlens_q->Shape()[0] - 1;

  ORT_ENFORCE(batch_size > 0);
  ORT_ENFORCE((head_size % 8 == 0) && (head_size <= 128));

  // CHECK_SHAPE(q, total_q, num_heads, head_size);
  // CHECK_SHAPE(k, total_k, num_heads, head_size);
  // CHECK_SHAPE(v, total_k, num_heads, head_size);
  // CHECK_SHAPE(out, total_q, num_heads, head_size);

  // CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  // CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  bool loop = false;
  int max_seqlen_k = get_max_seqlen_k(max_seqlen_k_, head_size, loop);
  int max_seqlen_q = get_max_seqlen_q(max_seqlen_q_);

  // auto o = torch::empty({ total_q , num_heads, head_size }, opts);
  float* o_tmp_buffer = nullptr;
  float* softmax_lse_buffer = nullptr;
  if (loop) {
    o_tmp_buffer = reinterpret_cast<float*>(workspace);
    // TODO: alignment
    softmax_lse_buffer = o_tmp_buffer + total_q * num_heads * head_size;
  }

  if (zero_tensors) {
    //out.zero_();

    // volatile union float_int {
    //   unsigned int i;
    //   float f;
    // } x;
    // x.f = -std::numeric_limits<float>::infinity();
    // // cuMemsetD32(reinterpret_cast<CUdeviceptr>(softmax_lse_buffer), x.i, static_cast<size_t>(batch_size) * num_heads * max_seqlen_q);
  }

  set_params_fprop(launch_params.params,
                   batch_size,
                   max_seqlen_q,
                   max_seqlen_k,
                   num_heads,
                   head_size,
                   q, k, v, out,
                   cu_seqlens_q,
                   cu_seqlens_k,
                   o_tmp_buffer,
                   nullptr,
                   softmax_lse_buffer,
                   softmax_scale,
                   is_causal,
                   num_splits);

  run_fmha_fwd(launch_params);
}
