// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/math/bias_softmax.h"
#include "contrib_ops/cuda/math/bias_softmax_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BiasSoftmax,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    BiasSoftmax);

Status BiasSoftmax::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const TensorShape& X_shape = X->Shape();
  const TensorShape& B_shape = B->Shape();
  Tensor* Y = ctx->Output(0, X_shape);

  const int softmax_axis = static_cast<int>(HandleNegativeAxis(softmax_axis_, X_shape.NumDimensions()));
  const int N = static_cast<int>(X_shape.SizeToDimension(softmax_axis));
  const int D = static_cast<int>(X_shape.SizeFromDimension(softmax_axis));

  const int broadcast_axis = static_cast<int>(HandleNegativeAxis(broadcast_axis_, X_shape.NumDimensions()));
  const int broadcast_size = N / static_cast<int>(X_shape.SizeToDimension(broadcast_axis));

  const size_t elem_size = X->DataType()->Size();
  utils::MLTypeCallDispatcher<double, float, MLFloat16> t_disp(X->GetElementType());

  if (D <= 1024 && D * elem_size <= 4096) {
    // expect thread blocks can fill SM at high occupancy without overflowing registers
    t_disp.Invoke<DispatchBiasSoftmaxForward>(Stream(), Y, X, B, D, N, D, broadcast_size);
  } else {
    // need to fallback to add kernel + CUDA DNN library softmax call :/
    ORT_RETURN_IF_ERROR((t_disp.InvokeRet<Status, DispatchBiasSoftMaxForwardViaDnnLibrary>(
        Stream(), CudnnHandle(), D, N, broadcast_axis, softmax_axis, X_shape, X, B_shape, B, Y)));
  }

  return Status::OK();
}

template <typename T>
void DispatchBiasSoftmaxForward<T>::operator()(cudaStream_t stream, Tensor* output, const Tensor* input,
                                               const Tensor* input_bias, int element_count, int batch_count,
                                               int batch_stride, int bias_broadcast_size_per_batch) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto* input_data = reinterpret_cast<const CudaT*>(input->template Data<T>());
  const auto* bias_data = reinterpret_cast<const CudaT*>(input_bias->template Data<T>());
  auto* output_data = reinterpret_cast<CudaT*>(output->template MutableData<T>());
  DispatchBiasSoftmaxForwardImpl<CudaT>(stream, output_data, input_data, bias_data, element_count, batch_count,
                                        batch_stride, bias_broadcast_size_per_batch);
}

template <typename T>
Status DispatchBiasSoftMaxForwardViaDnnLibrary<T>::operator()(cudaStream_t stream, cudnnHandle_t cudaDnnHandle,
                                                              int element_count, int batch_count, int broadcast_axis,
                                                              int softmax_axis, const onnxruntime::TensorShape& X_shape,
                                                              const onnxruntime::Tensor* X,
                                                              const onnxruntime::TensorShape& B_shape,
                                                              const onnxruntime::Tensor* B, onnxruntime::Tensor* Y) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto* X_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  const auto* B_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
  auto* Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
  BinaryElementwisePreparation p;
  p.BinaryElementwiseBroadcastPrepareHelper(X_shape, B_shape, X_shape);
  return DispatchBiasSoftMaxForwardViaDnnLibraryImpl<CudaT>(
      stream, cudaDnnHandle, element_count, batch_count, broadcast_axis, softmax_axis, X_data, B_data, Y_data, p.args);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
