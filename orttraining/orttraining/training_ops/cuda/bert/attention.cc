// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/bert/attention.h"

#include "orttraining/training_ops/cuda/bert/fmha/attention_impl.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kAlignLSE = 32;

void GetBiasStrides(MemEfficientAttentionParams& params, const TensorShape& bias_shape) {
  // bias is broadcasted to [b, num_heads, m, n].
  int pos = static_cast<int>(bias_shape.NumDimensions()) - 1;
  int64_t running_stride = 1;
  if (pos >= 0) {
    running_stride *= bias_shape[pos];
    --pos;
  }
  params.bias_stride_m = pos >= 0 && bias_shape[pos] == params.m ? running_stride : 0;
  if (pos >= 0) {
    running_stride *= bias_shape[pos];
    --pos;
  }
  params.bias_stride_h = pos >= 0 && bias_shape[pos] == params.num_heads ? running_stride : 0;
  if (pos >= 0) {
    running_stride *= bias_shape[pos];
    --pos;
  }
  params.bias_stride_b = pos >= 0 && bias_shape[pos] == params.b ? running_stride : 0;
}

}  // namespace

ONNX_OPERATOR_KERNEL_EX(AttentionTraining, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 4)
                            .InputMemoryType(OrtMemTypeCPUInput, 5)
                            .OutputMemoryType(OrtMemTypeCPUOutput, 2)
                            .OutputMemoryType(OrtMemTypeCPUOutput, 3)
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float>()),
                        AttentionTraining);

ONNX_OPERATOR_KERNEL_EX(AttentionTrainingGrad, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .InputMemoryType(OrtMemTypeCPUInput, 7)
                            .InputMemoryType(OrtMemTypeCPUInput, 8)
                            .InputMemoryType(OrtMemTypeCPUInput, 9)
                            .InputMemoryType(OrtMemTypeCPUInput, 10)
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float>()),
                        AttentionTrainingGrad);

Status AttentionTraining::ComputeInternal(OpKernelContext* context) const {
  const Tensor* q_tensor = context->Input<Tensor>(0);
  const Tensor* k_tensor = context->Input<Tensor>(1);
  const Tensor* v_tensor = context->Input<Tensor>(2);
  const Tensor* bias_tensor = context->Input<Tensor>(3);
  const Tensor* scale_tensor = context->Input<Tensor>(4);
  const Tensor* ratio_tensor = context->Input<Tensor>(5);

  const TensorShape& q_shape = q_tensor->Shape();
  const TensorShape& k_shape = k_tensor->Shape();
  const TensorShape& v_shape = v_tensor->Shape();

  ORT_ENFORCE(q_shape.NumDimensions() == 4 && k_shape.NumDimensions() == 4 && v_shape.NumDimensions() == 4);
  // Batch sizes.
  ORT_ENFORCE(q_shape[0] == k_shape[0] && q_shape[0] == v_shape[0]);
  // Sequence length.
  ORT_ENFORCE(k_shape[1] == v_shape[1]);
  // Num heads.
  ORT_ENFORCE(q_shape[2] == k_shape[2] && q_shape[2] == v_shape[2]);
  // Embedding per head.
  ORT_ENFORCE(q_shape[3] == k_shape[3]);

  MemEfficientAttentionFwdParams params;
  params.is_half = q_tensor->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  params.b = q_shape[0];
  params.m = q_shape[1];
  params.n = k_shape[1];
  params.num_heads = q_shape[2];
  params.k = q_shape[3];
  params.kv = v_shape[3];

  params.scale = scale_tensor ? *(scale_tensor->Data<float>()) : 1.0f / std::sqrt(static_cast<float>(params.k));
  params.p = ratio_tensor ? *(ratio_tensor->Data<float>()) : default_ratio_;
  ORT_ENFORCE(params.p >= 0.0f && params.p < 1.0f, "ratio should be in [0, 1) range.");

  if (params.p > 0.0f) {
    PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();
    auto seed_offset =
        generator.NextPhiloxSeeds(static_cast<uint64_t>(params.b * params.num_heads * params.m * params.n));
    params.seed = seed_offset.first;
    params.offset = seed_offset.second;
    Tensor* seed_tensor = context->Output(2, {});
    Tensor* offset_tensor = context->Output(3, {});
    ORT_ENFORCE(seed_tensor && offset_tensor);
    *(seed_tensor->MutableData<int64_t>()) = static_cast<int64_t>(params.seed);
    *(offset_tensor->MutableData<int64_t>()) = static_cast<int64_t>(params.offset);
  }

  Tensor* output_tensor = context->Output(0, {params.b, params.m, params.num_heads, params.kv});
  // LogSumExp tensor's shape is [b, num_heads, ceil_div(m, kAlignLSE) * kAlignLSE].
  Tensor* log_sum_exp_tensor =
      context->Output(1, {params.b, params.num_heads, (params.m + kAlignLSE - 1) / kAlignLSE * kAlignLSE});

  params.query = q_tensor->DataRaw();
  params.key = k_tensor->DataRaw();
  params.value = v_tensor->DataRaw();

  params.bias = nullptr;
  if (bias_tensor) {
    params.bias = bias_tensor->DataRaw();
    GetBiasStrides(params, bias_tensor->Shape());
  }

  params.output_fwd = output_tensor->MutableDataRaw();
  params.log_sum_exp_fwd = log_sum_exp_tensor->MutableData<float>();
  params.kernel = this;
  params.stream = context->GetComputeStream();
  return MemEfficentAttentionForward(params);
}

Status AttentionTrainingGrad::ComputeInternal(OpKernelContext* context) const {
  const Tensor* dy_tensor = context->Input<Tensor>(0);
  const Tensor* y_tensor = context->Input<Tensor>(1);
  const Tensor* q_tensor = context->Input<Tensor>(2);
  const Tensor* k_tensor = context->Input<Tensor>(3);
  const Tensor* v_tensor = context->Input<Tensor>(4);
  const Tensor* log_sum_exp_tensor = context->Input<Tensor>(5);
  const Tensor* bias_tensor = context->Input<Tensor>(6);
  const Tensor* scale_tensor = context->Input<Tensor>(7);
  const Tensor* ratio_tensor = context->Input<Tensor>(8);
  const Tensor* seed_tensor = context->Input<Tensor>(9);
  const Tensor* offset_tensor = context->Input<Tensor>(10);

  const TensorShape& dy_shape = dy_tensor->Shape();
  const TensorShape& y_shape = y_tensor->Shape();
  const TensorShape& q_shape = q_tensor->Shape();
  const TensorShape& k_shape = k_tensor->Shape();
  const TensorShape& v_shape = v_tensor->Shape();
  const TensorShape& log_sum_exp_shape = log_sum_exp_tensor->Shape();

  // ndim.
  ORT_ENFORCE(dy_shape.NumDimensions() == 4 && y_shape.NumDimensions() == 4 && q_shape.NumDimensions() == 4 &&
              k_shape.NumDimensions() == 4 && v_shape.NumDimensions() == 4 && log_sum_exp_shape.NumDimensions() == 3);

  // Batch sizes.
  ORT_ENFORCE(dy_shape[0] == y_shape[0] && dy_shape[0] == q_shape[0] && dy_shape[0] == k_shape[0] &&
              dy_shape[0] == v_shape[0] && dy_shape[0] == log_sum_exp_shape[0]);

  // Sequence length.
  ORT_ENFORCE(dy_shape[1] == y_shape[1] && dy_shape[1] == q_shape[1] && k_shape[1] == v_shape[1]);

  // Num heads.
  ORT_ENFORCE(dy_shape[2] == y_shape[2] && dy_shape[2] == q_shape[2] && dy_shape[2] == k_shape[2] &&
              dy_shape[2] == v_shape[2]);

  // Embedding per head.
  ORT_ENFORCE(dy_shape[3] == y_shape[3] && dy_shape[3] == v_shape[3] && q_shape[3] == k_shape[3]);

  MemEfficientAttentionBwdParams params;
  params.is_half = q_tensor->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  params.b = q_shape[0];
  params.m = q_shape[1];
  params.n = k_shape[1];
  params.num_heads = q_shape[2];
  params.k = q_shape[3];
  params.kv = v_shape[3];
  params.lse_stride = log_sum_exp_shape[2];

  params.scale = scale_tensor ? *(scale_tensor->Data<float>()) : 1.0f / std::sqrt(static_cast<float>(params.k));
  params.p = ratio_tensor ? *(ratio_tensor->Data<float>()) : default_ratio_;
  ORT_ENFORCE(params.p >= 0.0f && params.p < 1.0f, "ratio should be in [0, 1) range.");

  if (params.p > 0.0f) {
    ORT_ENFORCE(seed_tensor && offset_tensor);
    params.seed = static_cast<uint64_t>(*(seed_tensor->Data<int64_t>()));
    params.offset = static_cast<uint64_t>(*(offset_tensor->Data<int64_t>()));
  }

  Tensor* grad_query_tensor = context->Output(0, q_shape);
  Tensor* grad_key_tensor = context->Output(1, k_shape);
  Tensor* grad_value_tensor = context->Output(2, v_shape);
  Tensor* grad_bias_tensor = nullptr;
  params.bias = nullptr;
  if (bias_tensor) {
    params.bias = bias_tensor->DataRaw();
    const TensorShape& bias_shape = bias_tensor->Shape();
    GetBiasStrides(params, bias_shape);
    grad_bias_tensor = context->Output(3, bias_shape);
  }

  params.grad_output = dy_tensor->DataRaw();
  params.output_bwd = y_tensor->DataRaw();
  params.query = q_tensor->DataRaw();
  params.key = k_tensor->DataRaw();
  params.value = v_tensor->DataRaw();
  params.log_sum_exp_bwd = log_sum_exp_tensor->Data<float>();
  params.grad_query = grad_query_tensor->MutableDataRaw();
  params.grad_key = grad_key_tensor->MutableDataRaw();
  params.grad_value = grad_value_tensor->MutableDataRaw();
  params.grad_bias = grad_bias_tensor ? grad_bias_tensor->MutableDataRaw() : nullptr;
  params.kernel = this;
  params.stream = context->GetComputeStream();
  return MemEfficentAttentionBackward(params);
}

}  // namespace cuda
}  // namespace onnxruntime
