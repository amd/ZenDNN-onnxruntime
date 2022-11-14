// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/fused_elementwise.h"

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/math/fused_elementwise_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

namespace {

static OpType GetOpType(const std::string& op_type) {
  if (op_type == "Add") {
    return OpType::Add;
  } else if (op_type == "Sub") {
    return OpType::Sub;
  } else if (op_type == "Mul") {
    return OpType::Mul;
  } else if (op_type == "Div") {
    return OpType::Div;
  } else {
    ORT_THROW("Unsupported op type: ", op_type);
  }
}

template <typename T>
struct DispatchFusedElementwiseImpl {
  void operator()(cudaStream_t stream, int num_ops, Tensor* p_output, const Tensor* p_input1, const Tensor* p_input2,
                  const Tensor* p_input3, const Tensor* p_input4, OpType op1_type, OpType op2_type, OpType op3_type,
                  int N) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    bool is_input1_scalar = false;
    bool is_input2_scalar = p_input2->Shape().Size() == 1;
    bool is_input3_scalar = p_input3->Shape().Size() == 1;
    bool is_input4_scalar = num_ops == 3 ? p_input4->Shape().Size() == 1 : false;
    CudaT* p_output_data = reinterpret_cast<CudaT*>(p_output->MutableData<T>());
    const CudaT* p_input1_data = reinterpret_cast<const CudaT*>(p_input1->Data<T>());
    const CudaT* p_input2_data = reinterpret_cast<const CudaT*>(p_input2->Data<T>());
    const CudaT* p_input3_data = reinterpret_cast<const CudaT*>(p_input3->Data<T>());
    const CudaT* p_input4_data = num_ops == 3 ? reinterpret_cast<const CudaT*>(p_input4->Data<T>()) : nullptr;
    FusedElementwiseImpl<CudaT>(stream, num_ops, p_output_data, p_input1_data, p_input2_data, p_input3_data,
                                p_input4_data, is_input1_scalar, is_input2_scalar, is_input3_scalar, is_input4_scalar,
                                op1_type, op2_type, op3_type, N);
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    FusedElementwise, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16>()), FusedElementwise);

Status FusedElementwise::ComputeInternal(OpKernelContext* ctx) const {
  int num_ops = static_cast<int>(op_types_.size());
  const Tensor* input1 = ctx->Input<Tensor>(0);
  const TensorShape& shape = input1->Shape();
  const Tensor* input2 = ctx->Input<Tensor>(1);
  const Tensor* input3 = ctx->Input<Tensor>(2);
  const Tensor* input4 = num_ops == 3 ? ctx->Input<Tensor>(3) : nullptr;
  OpType op1_type = GetOpType(op_types_[0]);
  OpType op2_type = GetOpType(op_types_[1]);
  OpType op3_type = num_ops == 3 ? GetOpType(op_types_[2]) : OpType::Add;
  Tensor* output = ctx->Output(0, shape);
  utils::MLTypeCallDispatcher<float, MLFloat16> t_disp(input1->GetElementType());
  t_disp.Invoke<DispatchFusedElementwiseImpl>(Stream(), num_ops, output, input1, input2, input3, input4, op1_type,
                                              op2_type, op3_type, static_cast<int>(shape.Size()));
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
