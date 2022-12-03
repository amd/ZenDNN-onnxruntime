#include "core/providers/cuda/cuda_common.h"
#include "core/framework/float16.h"

size_t get_softmax_lse_size(int max_seqlen_q_, int batch_size, int num_heads);

size_t get_o_tmp_size(int max_seqlen_k_, int total_q, int num_heads, int head_size, int v_head_size);

void fmha_forward(const cudaDeviceProp& dprops,
                  cudaStream_t stream,
                  void* q,    // shape: (total_q, num_heads, head_size)
                  void* k,    // shape: (total_k, num_heads, head_size)
                  void* v,    // shape: (total_k, num_heads, head_size)
                  void* out,  // shape: (total_q, num_heads, head_size)
                  int32_t* cu_seqlens_q,     // shape: (batch_size + 1)
                  int32_t* cu_seqlens_k,     // shape: (batch_size + 1)
                  void* softmax_lse_buffer,
                  void* o_tmp_buffer,
                  const int batch_size,
                  const int num_heads,
                  const int head_size,
                  const int v_head_size,
                  const int total_q,
                  const int max_seqlen_q_,
                  const int max_seqlen_k_,
                  const float softmax_scale,
                  const bool zero_tensors,
                  const bool is_causal,
                  const int num_splits);
