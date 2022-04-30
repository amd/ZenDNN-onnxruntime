// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xnnpack_execution_provider.h"
#include "detail/utils.h"
#include "detail/op_support_checker.h"

#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/session_options.h"

#include <xnnpack.h>

namespace onnxruntime {

namespace xnnpack {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

// NCHW stubs
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 1, 10, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 11, Conv);

// NHWC 'real' kernels
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 1, 10, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 11, Conv);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      // orig NCHW ops with dummy kernel
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 11, Conv)>,
      // 'real' NHWC kernels
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 1, 10, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 11, Conv)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

  return kernel_registry;
}

}  // namespace xnnpack

using namespace xnnpack;

XnnpackExecutionProvider::XnnpackExecutionProvider(const XnnpackExecutionProviderInfo& info)
    : IExecutionProvider{kXnnpackExecutionProvider, true},
      session_options_{info.session_options} {
  // TODO: Could/should we provide our default CPU allocator to this call in an adapter?
  // If so, probably need to move it out of the ctor
  xnn_status st = xnn_initialize(nullptr);
  if (st != xnn_status_success) {
    ORT_THROW("XNNPACK initialization failed with status ", st);
  }

  // TODO: Allocation planner calls GetAllocator for the individual EP. It would be better if it goes through
  // the session state to get the allocator so it's per-device (or for the allocation planner to try the EP first
  // and fall back to using session state next by passing in a functor it can use to call SessionState::GetAllocator).
  // That way we only need one allocator per-device unless the EP needs/wants to override any previously registered
  // allocator.

  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(kXnnpackExecutionProvider,
                                                            OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

std::vector<std::unique_ptr<ComputeCapability>> XnnpackExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> capabilities;

  std::shared_ptr<KernelRegistry> registry = GetKernelRegistry();
  std::unordered_set<const Node*> supported_nodes;
  NodeSupportChecker checker{graph, supported_nodes};

  // L2 optimizations include fusing Conv+Activation, which we may do with Conv+Clip.
  // check that the session options are available, and if so whether L2 optimizations are enabled.
  // If they're not available we can't assume the fusion will occur, so we can't take the activation node.
  bool l2_optimizations_enabled = session_options_ &&
                                  session_options_->graph_optimization_level >= TransformerLevel::Level2;

  // handle any nodes we have a static kernel for.
  const auto& nodes = graph.Nodes();

  for (auto iter = nodes.cbegin(), end = nodes.cend(); iter != end; ++iter) {
    bool request_node = false;
    const Node& node = *iter;

    if (node.GetExecutionProviderType() == "") {
      // unassigned node. check if we have a kernel registration for it.
      if (KernelRegistry::HasImplementationOf(*registry, node, Type())) {
        // we have a kernel registration for the operator, and any type constraints that have been satisfied.
        // if there are additional constraints such as checking values of attributes etc. those should
        // be checked here.
        request_node = checker.IsNodeSupported(node, /*matched_kernel*/ true);
      } else {
        // see if it's an activation we can fuse with a node we support
        if (l2_optimizations_enabled) {
          request_node = checker.IsNodeSupported(node, /*matched_kernel*/ false);
        }
      }
    } else if (node.GetExecutionProviderType() == Type()) {
      if (node.Op() == nullptr) {
        // node is assigned to us but the operator has no schema.
        //
        // it must have come from the NHWC transform if it is a layout sensitive op because...
        //
        // Graph::Resolve is not called after layout transform so that layout transform works in the minimal build.
        // Due to that a node we asked for that just had the layout changed to NHWC will have a nullptr for Op().
        //     Side note: Layout transform maintains edges and update shapes, so the graph should still be valid
        //                and the node should have valid type/shape info.
        //
        // We can't do a kernel registry lookup here as that requires the schema returned by Op().
        //     Side note: Whilst we _could_ update GraphPartitioner's GetCapabilityForEP implementation to call
        //                Graph::SetOpSchemaFromRegistryForNode to set Op() if a schema for the NHWC op existed,
        //                we can only do that in a full build, so it's not a general purpose solution.
        //
        // However, we shouldn't need to do the kernel registry lookup:
        //   The sequence of calls is
        //      GetCapability ->
        //      layout transform for layout sensitive ops in that set of nodes ->
        //      GetCapability
        //
        //   Any node that does NOT have an Op() that is assigned to us can only be seen in the second call to
        //   GetCapability, and should only be a layout sensitive op.
        //
        // So provided we only returned layout sensitive nodes in the first call to GetCapability for which we have
        // an NHWC kernel, we can infer that we support the replacement node.
        //
        // IMPORTANT NOTE: We will have a hard requirement on the new approach to enable kernel matching at runtime
        //                 in a minimal build.
        ORT_ENFORCE(node.Domain() == kMSInternalNHWCDomain,
                    "Node is assigned to us but is not the NHWC version of a node we originally asked for.");
      } else {
        // node we selected in the first call to GetCapability. no need to check if we have a kernel again.
      }

      request_node = true;
    } else {
      // node belongs to another EP
      continue;
    }

    if (request_node) {
      // create a ComputeCapability for the individual node.
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      capabilities.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  // finding nodes to compile can be inserted here and added to the ComputeCapability instances returned.
  // GraphPartitioner will can handle a mix of static and compiled kernels.

  return capabilities;
}

std::shared_ptr<KernelRegistry> XnnpackExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> registry = xnnpack::RegisterKernels();
  return registry;
}

XnnpackExecutionProvider::~XnnpackExecutionProvider() {
  xnn_deinitialize();
}

}  // namespace onnxruntime
