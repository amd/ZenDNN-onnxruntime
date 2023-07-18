/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*******************************************************************************
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*******************************************************************************/

#pragma once

#include <map>
#include <list>
#include <memory>
#include <memory.h>

#include "core/providers/zendnn/zendnn_execution_provider_info.h"
#include "core/providers/zendnn/zendnn_threadpool.h"
#include "core/providers/zendnn/zendnn_op_manager.h"
#include "core/providers/zendnn/subgraph/zendnn_subgraph.h"
#include "core/providers/zendnn/subgraph/zendnn_subgraph_primitive.h"

namespace onnxruntime {

// Logical device representation.
class ZendnnExecutionProvider : public IExecutionProvider {
  public:
    explicit ZendnnExecutionProvider(const ZendnnExecutionProviderInfo &info);
    virtual ~ZendnnExecutionProvider();

    std::vector<std::unique_ptr<ComputeCapability>>
            GetCapability(const onnxruntime::GraphViewer &graph,
                          const IKernelLookup & /*kernel_lookup*/) const override;

    common::Status Compile(const std::vector<FusedNodeAndGraph>
                           &fused_nodes_and_graphs,
                           std::vector<NodeComputeInfo> &node_compute_funcs) override;

  private:
    ZendnnExecutionProviderInfo info_;
    // ZendnnOpManager contains information about supported Zendnn Operators
    ZendnnOpManager opManager_;
    std::unordered_map<std::string, std::unique_ptr<ort_zendnn::ZendnnSubgraph>>
            subgraphs_;
    std::unordered_map<std::string, std::unique_ptr<ort_zendnn::ZendnnSubgraphPrimitive>>
            subgraph_primitives_;
    std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer
                                     &graph_viewer) const;
    // dump subgraphs to onnx format for debugging purpose
    bool dump_subgraphs_ = false;
    bool debug_log_ = false;
    // enable fusion by default
    bool enable_fusion_ = true;
    AllocatorPtr mem_allocator;
    /* Enable inplace concat by setting the environment flag expicitely
     * Currently, we support following optimizations..
     */
    bool enable_inplace_optimizations_ = false;
};

}  // namespace onnxruntime
