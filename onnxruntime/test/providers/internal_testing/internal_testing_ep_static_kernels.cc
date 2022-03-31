// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "internal_testing_ep_static_kernels.h"
#include "core/framework/utils.h"

namespace onnxruntime {
namespace internal_testing_ep {

// can't use 'utils::kInternalTestingExecutionProvider' in the macro so redefine here to a name without '::'
constexpr const char* internal_testing_ep = utils::kInternalTestingExecutionProvider;

// register the 'real' NHWC kernels
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Conv, kMSInternalNHWCDomain, 1, 10, internal_testing_ep,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  Conv);

ONNX_OPERATOR_KERNEL_EX(Conv, kMSInternalNHWCDomain, 11, internal_testing_ep,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        Conv);

// and register matching stubs in the NCHW domain. We match against the stubs in GetCapability, after which the
// GraphPartitioner should convert to NHWC format. The updated node will be matched in session state finalization.
// NOTE: This dual registration is only required for the small subset of layout sensitive ops.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Conv, kOnnxDomain, 1, 10, internal_testing_ep,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  InvalidNchwKernel);

ONNX_OPERATOR_KERNEL_EX(Conv, kOnnxDomain, 11, internal_testing_ep,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        InvalidNchwKernel);

}  // namespace internal_testing_ep
}  // namespace onnxruntime
