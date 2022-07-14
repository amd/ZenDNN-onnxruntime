// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/copy.h"

#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/tensor/copy_impl.h"

namespace onnxruntime {
namespace cuda {

Status StridedCopyTensor(cudaStream_t stream, const Tensor& src, Tensor& dst) {
  if (src.DataRaw() == dst.DataRaw()) return Status::OK();
  TensorShapeVector src_strides = TensorPitches(src.Shape());
  TensorShapeVector dst_strides = TensorPitches(dst.Shape());
#ifdef ENABLE_TRAINING
  src_strides = ToShapeVector(src.Strides());
  dst_strides = ToShapeVector(dst.Strides());
#endif
  if (src_strides == dst_strides) {
    CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(dst.MutableDataRaw(), src.DataRaw(), src.SizeInBytes(), cudaMemcpyDeviceToDevice, stream));
    return Status::OK();
  }

  BinaryElementwisePreparation p;
  p.lhs_tensor = &src;
  p.rhs_tensor = &dst;
  p.BinaryElementwiseBroadcastPrepareHelper(src.Shape(), dst.Shape(), dst.Shape());
  switch (src.DataType()->Size()) {
#define CASE_DATATYPE(type)                                                                   \
  case sizeof(type): {                                                                        \
    typedef typename ToCudaType<type>::MappedType CudaT;                                      \
    StridedCopyImpl<CudaT>(stream, reinterpret_cast<const CudaT*>(src.template Data<type>()), \
                           dst.template MutableData<type>(), p.args);                         \
  } break
    CASE_DATATYPE(int8_t);
    CASE_DATATYPE(int16_t);
    CASE_DATATYPE(int32_t);
    CASE_DATATYPE(int64_t);
    default:
      ORT_THROW("Unsupported element size by StridedCopyTensor.");
#undef CASE_DATATYPE
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
