#include "core/providers/cuda/cuda_common.h"
#include "core/framework/float16.h"

size_t get_flash_attention_workspace_size(int max_seqlen_q_, int max_seqlen_k_, int batch_size, int total_q, int num_heads, int head_size);

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
                  const int num_splits);
