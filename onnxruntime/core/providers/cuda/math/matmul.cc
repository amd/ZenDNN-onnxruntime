// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/tensor/scatter_nd_impl.h"
#include "core/framework/float16.h"
#include <fstream>

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      1, 8,                                                       \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      9, 12,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MatMul,                                                     \
      kOnnxDomain,                                                \
      13,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

// StridedBatchedGemm can be used for the following GEMM computation
// C[pnm] = A[pnk]*B[km] or C[pnm] = A[pnk]*B[pkm]
static bool CanUseStridedBatchedGemm(const TensorShape& left_shape, const TensorShape& right_shape,
                                     bool transa, bool transb, bool trans_batch_a, bool trans_batch_b,
                                     int64_t& stride_A, int64_t& stride_B, int64_t& stride_C, int64_t& batch_count) {
  size_t left_num_dims = left_shape.NumDimensions();
  size_t right_num_dims = right_shape.NumDimensions();

  if (!(left_num_dims >= 3 && right_num_dims >= 2)) {
    return false;
  }

  size_t left_leading_axis = trans_batch_a ? 0 : left_num_dims - 2;
  size_t right_leading_axis = trans_batch_b ? 0 : right_num_dims - 2;
  int64_t left_p = left_shape.SizeToDimension(left_num_dims - 2);
  if (trans_batch_a) {
    left_p = left_p * left_shape[left_num_dims - 2] / left_shape[0];
  }
  int64_t left_k = transa ? left_shape[left_leading_axis] : left_shape[left_num_dims - 1];

  if (right_num_dims >= 3) {
    int64_t right_p = right_shape.SizeToDimension(right_num_dims - 2);
    if (trans_batch_b) {
      right_p = right_p * right_shape[right_num_dims - 2] / right_shape[0];
    }
    if (left_p != right_p) {
      return false;
    }
  }

  int64_t right_k = transb ? right_shape[right_num_dims - 1] : right_shape[right_leading_axis];
  if (left_k != right_k) {
    return false;
  }

  int64_t n = transa ? left_shape[left_num_dims - 1] : left_shape[left_leading_axis];
  int64_t m = transb ? right_shape[right_leading_axis] : right_shape[right_num_dims - 1];
  stride_A = n * left_k / (trans_batch_a ? left_shape[0] : 1);
  stride_B = right_num_dims == 2 ? 0 : right_k * m / (trans_batch_b ? right_shape[0] : 1);
  stride_C = n * m;
  batch_count = left_p;
  return true;
}

template <typename T>
Status MatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  if (Node().Name() == "MatMul_688") {
    bool randomize_weights = ParseEnvironmentVariableWithDefault<bool>("ORT_RANDOMIZE_WEIGHTS", false);
    std::string file_prefix = "slow";

    if (randomize_weights) {
      file_prefix = "fast";
    }

    std::ofstream myfile_left;
    std::ofstream myfile_right;

    myfile_left.open(file_prefix + "_left.txt");
    myfile_right.open(file_prefix + "_right.txt");

    std::vector<uint16_t> host_left_X(left_X->Shape().Size(), 0);
    std::vector<uint16_t> host_right_Y(right_X->Shape().Size(), 0);

    cudaDeviceSynchronize();
    cudaMemcpy(host_left_X.data(), left_X->DataRaw(), left_X->SizeInBytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_right_Y.data(), right_X->DataRaw(), right_X->SizeInBytes(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    size_t counter = 0;
    for (const auto& e : host_left_X) {
      myfile_left << e;

      if (++counter != static_cast<size_t>(left_X->Shape().Size())) {
        myfile_left << std::endl;
      }
    }

    counter = 0;
    for (const auto& e : host_right_Y) {
      myfile_right << e;

      if (++counter != static_cast<size_t>(right_X->Shape().Size())) {
        myfile_right << std::endl;
      }
    }

    myfile_left.close();
    myfile_right.close();
  }

  // Ignore the transpose flag if rank of input being 1.
  // Be noted: numpy.transpose on vector does not change anything.
  bool transa = trans_A_;
  bool transb = trans_B_;
  if (left_X->Shape().NumDimensions() == 1) {
    transa = false;
  }
  if (right_X->Shape().NumDimensions() == 1) {
    transb = false;
  }

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape(), transa, transb, trans_batch_a_, trans_batch_b_, false));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const CudaT alpha = ToCudaType<T>::FromFloat(alpha_);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  cublasOperation_t transA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int lda = helper.Lda(transa);
  const int ldb = helper.Ldb(transb);
  const int ldc = helper.Ldc();
  int64_t stride_A, stride_B, stride_C, batch_count;
  auto& device_prop = GetDeviceProp();
  if (helper.OutputOffsets().size() == 1) {
    if (should_use_cublas_gemm_) {
      if (flush_denormals_to_zero_ || flush_all_to_zero_) {
        cudaDeviceSynchronize();

#ifdef _WIN32
        SubnormalFlush(Stream(),
                       const_cast<Tensor*>(left_X)->MutableDataRaw(),
                       768,
                       1,
                       128,
                       flush_all_to_zero_ ? 1 : 0);

        SubnormalFlush(Stream(),
                       const_cast<Tensor*>(right_X)->MutableDataRaw(),
                       static_cast<int>(right_X->Shape()[1]),
                       1,
                       static_cast<int>(right_X->Shape()[0]),
                       flush_all_to_zero_ ? 1 : 0);
#else
        SubnormalFlush(Stream(),
                       const_cast<Tensor*>(left_X)->MutableDataRaw(),
                       768,
                       32,
                       128,
                       flush_all_to_zero_ ? 1 : 0);

        SubnormalFlush(Stream(),
                       const_cast<Tensor*>(right_X)->MutableDataRaw(),
                       static_cast<int>(right_X->Shape()[1]),
                       1,
                       static_cast<int>(right_X->Shape()[0]),
                       flush_all_to_zero_ ? 1 : 0);
#endif

        /*

        // Flush sub-normals to zero
        std::vector<uint16_t> input_A(left_X->Shape().Size(), 0);
        std::vector<uint16_t> input_B(right_X->Shape().Size(), 0);

        cudaMemcpyAsync(input_A.data(), left_X->DataRaw(), left_X->SizeInBytes(), cudaMemcpyDeviceToHost, Stream());
        cudaMemcpyAsync(input_B.data(), right_X->DataRaw(), right_X->SizeInBytes(), cudaMemcpyDeviceToHost, Stream());

        cudaDeviceSynchronize();

        size_t subnormal_cnt_A = 0;
        for (size_t i = 0; i < static_cast<size_t>(left_X->Shape().Size()); ++i) {
          if ((input_A[i] & 0x7C00) == 0) {
            ++subnormal_cnt_A;
            //input_A[i] = 0;
          }
        }

        size_t subnormal_cnt_B = 0;
        for (size_t i = 0; i < static_cast<size_t>(right_X->Shape().Size()); ++i) {
          if ((input_B[i] & 0x7C00) == 0) {
            ++subnormal_cnt_B;
            //input_B[i] = 0;
          }
        }

        cudaMemcpyAsync(const_cast<Tensor*>(left_X)->MutableDataRaw(), input_A.data(), left_X->SizeInBytes(), cudaMemcpyHostToDevice, Stream());
        cudaMemcpyAsync(const_cast<Tensor*>(right_X)->MutableDataRaw(), input_B.data(), right_X->SizeInBytes(), cudaMemcpyHostToDevice, Stream());
         
         */
      }

      cudaDeviceSynchronize();

      auto start = high_resolution_clock::now();

      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          Base::CublasHandle(),
          transB,
          transA,
          static_cast<int>(helper.N()),
          static_cast<int>(helper.M()),
          static_cast<int>(helper.K()),
          &alpha,
          reinterpret_cast<const CudaT*>(right_X->Data<T>()),
          ldb,
          reinterpret_cast<const CudaT*>(left_X->Data<T>()),
          lda,
          &zero,
          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
          ldc,
          device_prop));

      cudaStreamSynchronize(Stream());

      auto stop = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(stop - start);

      std::cout << std::endl;
      //float frac = ((subnormal_cnt_A + subnormal_cnt_B) * 100.f) / (left_X->Shape().Size() + right_X->Shape().Size());
      if (Node().Name() == "MatMul_688") {
        std::cout << Node().Name() << " : " << duration.count() << std::endl;
      }
    } else {
      CUBLAS_RETURN_IF_ERROR(cublasLtMatmulHelper(
          CublasLtHandle(),
          transB,
          transA,
          static_cast<int>(helper.N()),
          static_cast<int>(helper.M()),
          static_cast<int>(helper.K()),
          &alpha,
          reinterpret_cast<const CudaT*>(right_X->Data<T>()),
          ldb,
          reinterpret_cast<const CudaT*>(left_X->Data<T>()),

          lda,
          &zero,
          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
          ldc,
          NULL, false,
          NULL, 0,
          Stream()));
    }

    return Status::OK();
  } else if (CanUseStridedBatchedGemm(left_X->Shape(), right_X->Shape(),
                                      transa, transb, trans_batch_a_, trans_batch_b_, stride_A, stride_B, stride_C, batch_count)) {
    CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(Base::CublasHandle(),
                                                          transB,
                                                          transA,
                                                          static_cast<int>(helper.N()),
                                                          static_cast<int>(helper.M()),
                                                          static_cast<int>(helper.K()),
                                                          &alpha,
                                                          reinterpret_cast<const CudaT*>(right_X->Data<T>()),
                                                          ldb,
                                                          stride_B,
                                                          reinterpret_cast<const CudaT*>(left_X->Data<T>()),
                                                          lda,
                                                          stride_A,
                                                          &zero,
                                                          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
                                                          ldc,
                                                          stride_C,
                                                          static_cast<int>(batch_count),
                                                          device_prop));

    return Status::OK();
  }

  // Fill offsets when needed.
  helper.FillOffsets();
  CudaAsyncBuffer<const CudaT*> left_arrays(this, helper.LeftOffsets().size());
  CudaAsyncBuffer<const CudaT*> right_arrays(this, helper.RightOffsets().size());
  CudaAsyncBuffer<CudaT*> output_arrays(this, helper.OutputOffsets().size());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(left_X->Data<T>()), helper.LeftOffsets(), left_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<const CudaT*>(right_X->Data<T>()), helper.RightOffsets(), right_arrays.CpuSpan());
  MatMulComputeHelper::OffsetToArrays(reinterpret_cast<CudaT*>(Y->MutableData<T>()), helper.OutputOffsets(), output_arrays.CpuSpan());
  ORT_RETURN_IF_ERROR(left_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(right_arrays.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_arrays.CopyToGpu());

  // note that onnxruntime OrtValue is row major, while cublas is column major,
  // so swap left/right operands
  CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
      Base::CublasHandle(),
      transB,
      transA,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &alpha,
      right_arrays.GpuPtr(),
      ldb,
      left_arrays.GpuPtr(),
      lda,
      &zero,
      output_arrays.GpuPtr(),
      ldc,
      static_cast<int>(helper.OutputOffsets().size()),
      device_prop));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
