// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_fwd.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/cuda/cuda_stream_handle.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace cuda {

uint32_t CheckSumImpl(cudaStream_t stream,
                      const void* input_data,
                      int nbytes);

// -----------------------------------------------------------------------
// Base class for CUDA kernels
// -----------------------------------------------------------------------
class CudaKernel : public OpKernel {
 public:
  //std::unique_ptr<uint32_t[]> output_crc32_;
  //uint32_t output_count_capacity_ = 0;
  //std::string kernel_name_;

  explicit CudaKernel(const OpKernelInfo& info)
      : OpKernel(info),
        // Is this OK to have a non-const execution provider?
        provider_(const_cast<CUDAExecutionProvider*>(static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()))) {
    //output_count_capacity_ = info.GetOutputCount();
    //output_crc32_ = std::make_unique<uint32_t[]>(output_count_capacity_);
    //memset(output_crc32_.get(), 0, output_count_capacity_ * sizeof(uint32_t));
    //kernel_name_ = info.node().Name();
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);
    // use this to precisely locate the node where CUDA failure comes from
      if (cudaSuccess != cudaDeviceSynchronize())
        __debugbreak();

    if (s.IsOK()) {
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err));
      }

      // XXX: Not thread-safe, experimental testing
      // The kernel may have variable number of outputs
      //auto cuda_stream = Stream(p_op_kernel_context);
      //const auto out_count = p_op_kernel_context->OutputCount();
      //for (int i = 0; i < out_count; ++i) {

      //  const auto& output = *p_op_kernel_context->Output<Tensor>(i);

      //  if (output.IsDataTypeString()) {
      //    continue;
      //  }

      //  const auto size = static_cast<int>(output.SizeInBytes());
      //  if (size > 0) {
      //    const auto* data = output.DataRaw();
      //    // Need to calculate Crc32 on device memory
      //    const uint32_t crc = CheckSumImpl(cuda_stream, data, size);
      //    if (output_crc32_[i] != 0) {
      //      if (output_crc32_[i] != crc) {
      //        const auto& logger = p_op_kernel_context->Logger();
      //        LOGS_DEFAULT(ERROR) << "##CUDA Output check:" << kernel_name_ << ": output " << i << " mismatch";
      //      }
      //    } else {
      //      output_crc32_[i] = crc;
      //    }
      //  }
      //}
    }
    return s;
  }

  virtual Status ComputeInternal(OpKernelContext* p_op_kernel_context) const = 0;

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes, onnxruntime::Stream* stream) const {
    return provider_->GetScratchBuffer<T>(count_or_bytes, stream, WaitCudaNotificationOnDevice);
  }

  // Different from GetScratchBuffer which use IAllocator::Alloc() to allocate memory,
  // this GetTransientScratchBuffer will call IAllocator::Reserve() to allocate memory.
  // IAllocator::Reserve() optionally implement some allocation logic that by-passes any arena-based
  // logic (or similar for different allocator) that may be housed in the Alloc() implementation.
  template <typename T>
  inline IAllocatorUniquePtr<T> GetTransientScratchBuffer(size_t count_or_bytes) const {
    return provider_->GetTransientScratchBuffer<T>(count_or_bytes);
  }

  inline void AddDeferredReleaseCPUPtr(void* p, onnxruntime::Stream* ort_stream) const {
    ORT_ENFORCE(ort_stream->GetDevice().Type() == OrtDevice::GPU);
    auto* cuda_ep_stream = static_cast<CudaStream*>(ort_stream);
    cuda_ep_stream->EnqueDeferredCPUBuffer(p);
  }

  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t count_or_bytes) const {
    return provider_->AllocateBufferOnCPUPinned<T>(count_or_bytes);
  }

  const cudaDeviceProp& GetDeviceProp() const { return provider_->GetDeviceProp(); }

  inline cudaStream_t Stream(OpKernelContext* ctx) const {
    auto* stream = ctx->GetComputeStream();
    return stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
  }

  inline cudnnHandle_t GetCudnnHandle(OpKernelContext* ctx) const {
    return GetCudnnHandle(static_cast<CudaStream*>(ctx->GetComputeStream()));
  }

  static inline cudnnHandle_t GetCudnnHandle(onnxruntime::CudaStream* stream) {
    return stream->cudnn_handle_;
  }

  inline cublasHandle_t GetCublasHandle(OpKernelContext* ctx) const {
    return GetCublasHandle(static_cast<CudaStream*>(ctx->GetComputeStream()));
  }

  static inline cublasHandle_t GetCublasHandle(onnxruntime::CudaStream* stream) {
    return stream->cublas_handle_;
  }

  tunable::CudaTuningContext* GetTuningContext() const {
    return static_cast<tunable::CudaTuningContext*>(provider_->GetTuningContext());
  }

  // To support cudaMemcpyAsync, the cpu memory should be allocated in pinned memory
  // and it can only be released after the copy has finished
  template <typename T>
  class CudaAsyncBuffer {
   public:
    CudaAsyncBuffer(const CudaKernel* op_kernel) : gpu_copy_(nullptr), count_(0), op_kernel_(op_kernel) {}

    CudaAsyncBuffer(const CudaKernel* op_kernel, size_t count) : CudaAsyncBuffer(op_kernel) {
      AllocCpuPtr(count);
    }

    CudaAsyncBuffer(const CudaKernel* op_kernel, const T& value, size_t count)
        : CudaAsyncBuffer(op_kernel, count) {
      T* p = CpuPtr();
      for (size_t i = 0; i != count; ++i) {
        *p++ = value;
      }
    }

    CudaAsyncBuffer(const CudaKernel* op_kernel, gsl::span<T const> vec) : CudaAsyncBuffer(op_kernel, vec.size()) {
      memcpy(CpuPtr(), vec.data(), vec.size() * sizeof(T));
    }

    void AllocCpuPtr(size_t count) {
      cpu_pinned_copy_ = op_kernel_->AllocateBufferOnCPUPinned<T>(count);
      if (cpu_pinned_copy_ == nullptr)
        throw std::runtime_error("alloc failed");
      count_ = count;
    }

    Status CopyToGpu(onnxruntime::Stream* stream) {
      if (cpu_pinned_copy_) {
        gpu_copy_ = op_kernel_->GetScratchBuffer<T>(count_, stream);
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(gpu_copy_.get(), cpu_pinned_copy_.get(), count_ * sizeof(T), cudaMemcpyHostToDevice,
                                             cuda_stream));
        op_kernel_->AddDeferredReleaseCPUPtr(cpu_pinned_copy_.release(), stream);
      }
      return Status::OK();
    }

    T* CpuPtr() const {
      return cpu_pinned_copy_.get();
    }

    gsl::span<T> CpuSpan() const {
      return gsl::span<T>(CpuPtr(), count_);
    }

    T* GpuPtr() const {
      return gpu_copy_.get();
    }

    size_t count() const {
      return count_;
    }

   protected:
    IAllocatorUniquePtr<T> gpu_copy_;
    IAllocatorUniquePtr<T> cpu_pinned_copy_;
    size_t count_;
    const CudaKernel* op_kernel_;
  };

  inline cublasHandle_t DefaultCublasHandle() const {
    return provider_->PerThreadDefaultCublasHandle();
  }

  inline cublasLtHandle_t CublasLtHandle() const {
    return provider_->PerThreadCublasLtHandle();
  }

  inline cudnnHandle_t DefaultCudnnHandle() const {
    return provider_->PerThreadDefaultCudnnHandle();
  }

 protected:
  template <typename T>
  inline const T* GetConstOnes(size_t count, cudaStream_t stream) const {
    return provider_->template GetConstOnes<T>(count, stream);
  }

  inline Status CopyTensor(const Tensor& src, Tensor& dst, onnxruntime::Stream& stream) const {
    auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(src.Location().device, dst.Location().device);
    return gpu_data_transfer->CopyTensorAsync(src, dst, stream);
  }

  inline int GetDeviceId() const { return provider_->GetDeviceId(); }

 private:
  CUDAExecutionProvider* provider_;
};

}  // namespace cuda
}  // namespace onnxruntime
