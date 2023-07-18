/*******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef _MSC_VER
    #pragma warning(disable : 4996)
#endif

#include "core/providers/zendnn/zendnn_execution_provider.h"

#include <fstream>
#include <iomanip>
#include <unordered_set>
#if defined(ZENDNN_OPENMP)
    #include <omp.h>
#endif  // defined(ZENDNN_OPENMP)

#include "core/platform/ort_mutex.h"
#include "core/providers/shared_library/provider_api.h"

#include "core/providers/zendnn/zendnn_fwd.h"
#include "core/providers/zendnn/zendnn_node_capability.h"
#include "core/providers/zendnn/subgraph/zendnn_subgraph_transformer.h"

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

#ifndef _WIN32
extern "C" {
    extern unsigned int graph_exe_count;
}
#endif

static long int num_sub_graphs;

constexpr const char *ZENDNN = "Zendnn";
constexpr const char *ZENDNN_CPU = "ZendnnCpu";

ZendnnExecutionProvider::ZendnnExecutionProvider(const
        ZendnnExecutionProviderInfo &info)
    : IExecutionProvider{onnxruntime::kZendnnExecutionProvider, true},
      info_(info) {
    InitProviderOrtApi();

    AllocatorCreationInfo default_memory_info( {
        [](int) {
            return onnxruntime::CreateCPUAllocator(OrtMemoryInfo(ZENDNN,
                                                   OrtAllocatorType::OrtDeviceAllocator));
        }},
    0, info.use_arena);

    AllocatorCreationInfo cpu_memory_info( {
        [](int) {
            return onnxruntime::CreateCPUAllocator(OrtMemoryInfo(ZENDNN_CPU,
                                                   OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0,
                                                   OrtMemTypeCPUOutput));
        }},
    0, info.use_arena);

    InsertAllocator(CreateAllocator(default_memory_info));
    InsertAllocator(CreateAllocator(cpu_memory_info));

    mem_allocator = GetAllocator(OrtMemTypeCPUOutput);

    // debug env variable to dump subgraphs
    const std::string dump_subgraphs_env =
        onnxruntime::GetEnvironmentVar("ORT_ZENDNN_DUMP_SUBGRAPHS");
    if (!dump_subgraphs_env.empty()) {
        dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
    }

    const std::string debug_log_env =
        onnxruntime::GetEnvironmentVar("ORT_ZENDNN_DEBUG_LOG");
    if (!debug_log_env.empty()) {
        debug_log_ = (std::stoi(debug_log_env) == 0 ? false : true);
    }

    const std::string fusion_env =
        onnxruntime::GetEnvironmentVar("ORT_ZENDNN_ENABLE_FUSION");
    if (!fusion_env.empty()) {
        enable_fusion_ = (std::stoi(fusion_env) == 0 ? false : true);
    }

    const std::string inplace_concat_env =
        onnxruntime::GetEnvironmentVar("ORT_ZENDNN_ENABLE_INPLACE_CONCAT");
    if (!inplace_concat_env.empty()) {
        enable_inplace_optimizations_ = (std::stoi(inplace_concat_env) == 0 ? false :
                                         true);
    }

    // Set the number of threads specified by the user
    // If provided arguments set them as the number of threads, else call
    // calc which usually = numcores
    auto num_threads = static_cast<int *>(info.threadpool_args);
    ORT_UNUSED_PARAMETER(num_threads);
#if defined(ZENDNN_OPENMP)
    // On Java we have limitations to the number of threads so let OpenMP decide
#if !defined(ZENDNN_JAVA)
    // If the user provided a value select between 3 cases
    // if num_threads < 0 OpenMP decides the number of threads
    if (num_threads != nullptr) {
        // The user provided a valid number of threads
        if (*num_threads > 0) {
            omp_set_num_threads(*num_threads);
        }
        else if (*num_threads == 0) {
            // If 0 then the user selected the default num_threads = num_physical_cores
            omp_set_num_threads(ZendnnCalcNumThreads());
        }
        // If no value was provided and the env var is not set, we define the number of threads to prevent oversubscription
    }
    else if (onnxruntime::GetEnvironmentVar("OMP_NUM_THREADS").empty()) {
        omp_set_num_threads(ZendnnCalcNumThreads());
    }
#else
    ORT_UNUSED_PARAMETER(num_threads);
#endif  // !defined(ZENDNN_JAVA)
    // Log the number of threads used
    LOGS_DEFAULT(INFO) << "Allocated " << omp_get_max_threads() <<
                       " OpenMP threads for ZenDNN ep\n";
#endif  // defined(ZENDNN_OPENMP)

}  // namespace onnxruntime

ZendnnExecutionProvider::~ZendnnExecutionProvider() {
}

std::vector<std::vector<NodeIndex>> ZendnnExecutionProvider::GetSupportedNodes(
const GraphViewer &graph_viewer) const {
    std::vector<std::vector<size_t>> supported_node_vecs;
    std::vector<size_t> supported_node_vec;

    std::unordered_map<std::string, int> all_nodes_count;
    std::unordered_map<std::string, int> supported_nodes_count;

    const auto &node_indices = graph_viewer.GetNodesInTopologicalOrder();
    for (size_t i = 0; i < node_indices.size(); i++) {
        auto node_idx = node_indices[i];
        const auto *node(graph_viewer.GetNode(node_idx));

        bool supported = opManager_.IsNodeSupported(node, graph_viewer);

        // update count
        if (debug_log_) {
            auto node_optype_ver = node->OpType() + "_" + std::to_string(
                                       node->SinceVersion());
            all_nodes_count[node_optype_ver]++;
            if (supported) {
                supported_nodes_count[node_optype_ver]++;
            }

            LOGS_DEFAULT(ERROR) << "Operator type: [" << node->OpType()
                                << "] index: [" << node_idx
                                << "] name: [" << node->Name()
                                << "] supported: [" << supported
                                << "]";
        }

        if (supported) {
            supported_node_vec.push_back(node_idx);
        }
        else {
            if (!supported_node_vec.empty()) {
                supported_node_vecs.push_back(supported_node_vec);
                supported_node_vec.clear();
            }
        }
    }

    if (!supported_node_vec.empty()) {
        supported_node_vecs.push_back(supported_node_vec);
    }

    // collect statistics and report
    if (debug_log_) {
        int all_counts = 0;
        int support_counts = 0;
        for (auto e : all_nodes_count) {
            auto optype_ver = e.first;
            auto all_count = e.second;
            auto support_count = supported_nodes_count[optype_ver];
            all_counts += all_count;
            support_counts += support_count;
            LOGS_DEFAULT(ERROR) << "Operator type: [" << optype_ver << "] coverage: " <<
                                support_count << ":" << all_count << " percentage: " << (float)support_count /
                                (float)all_count;
        }
        LOGS_DEFAULT(ERROR) << "Total coverge: " << support_counts << ":" << all_counts
                            << " percentage: " << (float)support_counts / (float)all_counts;
    }

    return supported_node_vecs;
}

std::vector<std::unique_ptr<ComputeCapability>>
        ZendnnExecutionProvider::GetCapability(
            const GraphViewer &graph_viewer,
const IKernelLookup & /*kernel_lookup*/) const {
    // follow from coreml ep's Getcapability

    std::vector<std::unique_ptr<ComputeCapability>> result;

    if (graph_viewer.IsSubgraph()) {
        return result;
    }

    const auto node_groups = GetSupportedNodes(graph_viewer);

    const auto &graph_output_list = graph_viewer.GetOutputs();
    std::unordered_set<const NodeArg *> graph_outputs(graph_output_list.cbegin(),
            graph_output_list.cend());

    size_t num_of_supported_nodes = 0;
    for (const auto &group : node_groups) {
        if (group.empty()) {
            continue;
        }

        num_of_supported_nodes += group.size();

        if (debug_log_) {
            LOGS_DEFAULT(ERROR) <<
                                "ZendnnExecutionProvider::GetCapability, current supported node group size: "
                                << group.size();
        }

        std::unordered_set<NodeIndex> node_set;
        node_set.reserve(group.size());
        for (const auto &index : group) {
            node_set.insert(index);
        }

        auto sub_graph = onnxruntime::IndexedSubGraph::Create();
        std::unordered_set<const NodeArg *> node_outputs;
        std::unordered_set<const NodeArg *> subgraph_inputs;
        std::unordered_set<const NodeArg *> subgraph_outputs;
        std::vector<const NodeArg *> ordered_subgraph_inputs;
        std::vector<const NodeArg *> ordered_subgraph_outputs;

        for (const auto &index : group) {
            sub_graph->Nodes().push_back(index);
            const auto *node = graph_viewer.GetNode(index);

            for (const auto *input : node->InputDefs()) {
                // if the node input was not produced by this subgraph, add it to the subgraph inputs.
                if (!input->Exists()) {
                    continue;
                }
                if (node_outputs.count(input) == 0) {
                    if (subgraph_inputs.count(input) == 0) {
                        subgraph_inputs.insert(input);
                        ordered_subgraph_inputs.push_back(input);
                    }
                }
            }

            const auto &output_defs = node->OutputDefs();
            for (const auto *output_def : output_defs) {
                if (!output_def->Exists()) {
                    continue;
                }
                node_outputs.insert(output_def);
                // if output is overall graph output we need to produce it.
                if (graph_outputs.count(output_def) != 0) {
                    ordered_subgraph_outputs.push_back(output_def);
                }
            }

            // if output connects to a node not in this subgraph we need to produce it
            for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd();
                    it != end; ++it) {
                if (node_set.count(it->GetNode().Index()) == 0) {
                    const auto *output_def = output_defs[it->GetSrcArgIndex()];
                    if (subgraph_outputs.count(output_def) == 0 &&
                            graph_outputs.count(output_def) == 0) {
                        subgraph_outputs.insert(output_def);
                        ordered_subgraph_outputs.push_back(output_def);
                    }
                }
            }
        }

        // Assign inputs and outputs to subgraph's meta_def
        HashValue model_hash;
        int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
        auto meta_def = ::onnxruntime::IndexedSubGraph_MetaDef::Create();
        meta_def->name() = "ZENDNN_" + std::to_string(model_hash) + "_" +
                           std::to_string(metadef_id);
        meta_def->domain() = kMSDomain;
        meta_def->since_version() = 1;
        meta_def->status() = ONNX_NAMESPACE::EXPERIMENTAL;

        for (const auto &input : ordered_subgraph_inputs) {
            meta_def->inputs().push_back(input->Name());
        }

        for (const auto &output : ordered_subgraph_outputs) {
            meta_def->outputs().push_back(output->Name());
        }

        sub_graph->SetMetaDef(std::move(meta_def));

        result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    }

    if (debug_log_) {
        float percent_zendnn = 100.0f * (static_cast<float>(num_of_supported_nodes) /
                                         static_cast<float>(graph_viewer.NumberOfNodes()));
        LOGS_DEFAULT(ERROR) << "ZendnnExecutionProvider::GetCapability,"
                            << " number of partitions supported by ZENDNN: " << result.size()
                            << " number of nodes in the graph: " << graph_viewer.NumberOfNodes()
                            << " number of nodes supported by ZENDNN: " << num_of_supported_nodes
                            << std::fixed << std::setprecision(2) << " (" << percent_zendnn << "%)";
    }

    if (dump_subgraphs_) {
        auto model = graph_viewer.CreateModel(*GetLogger());
        auto model_proto = model->ToProto();
        graph_viewer.ToProto(*model_proto->mutable_graph(), false, true);
        model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
        HashValue model_hash;
        int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
        std::fstream dump("ZENDNN_" + std::to_string(model_hash) + "_" + std::to_string(
                              metadef_id) + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
        model_proto->SerializeToOstream(dump);
    }
    num_sub_graphs = (long int)result.size();
    return result;
}

Status ZendnnExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>
                                        &fused_nodes_and_graphs,
                                        std::vector<NodeComputeInfo> &node_compute_funcs) {
    // follow from coreml ep's Compile
    for (auto &fused_node_graph : fused_nodes_and_graphs) {
        const GraphViewer &graph_body_viewer = fused_node_graph.filtered_graph;
        const Node &fused_node = fused_node_graph.fused_node;
        if (dump_subgraphs_) {
            auto model = graph_body_viewer.CreateModel(*GetLogger());
            auto model_proto = model->ToProto();
            graph_body_viewer.ToProto(*model_proto->mutable_graph(), false, true);
            model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
            std::fstream dump(fused_node.Name() + ".onnx",
                              std::ios::out | std::ios::trunc | std::ios::binary);
            model_proto->SerializeToOstream(dump);
        }

        // subgraph
        auto zendnn_subgraph = std::make_unique<ort_zendnn::ZendnnSubgraph>
                               (ort_zendnn::ZendnnSubgraph(graph_body_viewer));
        subgraphs_.emplace(fused_node.Name(), std::move(zendnn_subgraph));

        // apply transformation to subgraph
        if (enable_fusion_) {
            ort_zendnn::ZendnnGraphTransformer().Apply(*subgraphs_[fused_node.Name()].get(),
                    graph_body_viewer);
        }

        if (enable_inplace_optimizations_) {
            ort_zendnn::ZendnnGraphTransformer().OptimizeInplaceOps(
                *subgraphs_[fused_node.Name()].get());
        }

        // subgraph primitive
        auto zendnn_subgraph_primitive =
            std::make_unique<ort_zendnn::ZendnnSubgraphPrimitive>
            (*subgraphs_[fused_node.Name()].get(), mem_allocator);
        {
            const auto &input_defs = fused_node.InputDefs();
            std::vector<std::string> onnx_input_names(input_defs.size());
            for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
                onnx_input_names[i] = input_defs[i]->Name();
            }
            zendnn_subgraph_primitive->SetOrderedInputs(std::move(onnx_input_names));
        }
        {
            const auto &output_defs = fused_node.OutputDefs();
            std::vector<std::string> onnx_output_names(output_defs.size());
            for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
                onnx_output_names[i] = output_defs[i]->Name();
            }
            zendnn_subgraph_primitive->SetOrderedOutputs(std::move(onnx_output_names));
        }

        subgraph_primitives_.emplace(fused_node.Name(),
                                     std::move(zendnn_subgraph_primitive));

        NodeComputeInfo compute_info;

        compute_info.create_state_func = [&](ComputeContext* context,
        FunctionState* state) {
            *state = subgraph_primitives_[context->node_name].get();
            return 0;
        };

        compute_info.release_state_func = [](FunctionState state) {
            ORT_UNUSED_PARAMETER(state);
        };

        compute_info.compute_func = [](FunctionState state, const OrtApi * /* api */,
        OrtKernelContext* context) {
            Ort::KernelContext ctx(context);
            static long int num_invocations = 0;
            ort_zendnn::ZendnnSubgraphPrimitive *subgraph_primitive =
                reinterpret_cast<ort_zendnn::ZendnnSubgraphPrimitive *>(state);

            const size_t subgraph_num_inputs =
                subgraph_primitive->GetOrderedInputs().size();
            const size_t subgraph_num_outputs =
                subgraph_primitive->GetOrderedOutputs().size();
            const size_t context_num_outputs = ctx.GetOutputCount();
            const size_t context_num_inputs = ctx.GetInputCount();

            std::unordered_map<std::string, ort_zendnn::OnnxTensorData> inputs;
            inputs.reserve(subgraph_num_inputs);
            for (size_t i = 0; i < context_num_inputs; i++) {
                auto input_name = subgraph_primitive->GetOrderedInputs()[i];
                auto input_tensor = ctx.GetInput(i);
                auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
                auto shape = tensor_info.GetShape();
                // zendnn expectes non-const data
                void *inputBuffer = const_cast<void *>(input_tensor.GetTensorRawData());
                inputs.emplace(
                    input_name,
                ort_zendnn::OnnxTensorData{
                    ort_zendnn::OnnxTensorInfo{tensor_info.GetElementType(), shape},
                    inputBuffer,
                });
            }

            // lock each subgraph_primitive as multiple threads have shared memories
            {
                std::unique_lock<OrtMutex> lock(subgraph_primitive->GetMutex());
                subgraph_primitive->Compile(inputs);
                std::unordered_map<std::string, ort_zendnn::OnnxTensorData> outputs;
                outputs.reserve(subgraph_num_outputs);
                for (size_t i = 0; i < context_num_outputs; i++) {
                    auto output_name = subgraph_primitive->GetOrderedOutputs()[i];
                    auto output_md = subgraph_primitive->GetOutputInfo(output_name);
                    auto output_shape = output_md.dims();
                    // if an output is a scaler, zendnn internally uses tensor representation (eg, (1,1,...))
                    // but allocating an output with no shape instead of the equivalent tensorshape to avoid shape mismatch
                    if (subgraph_primitive->IsScalarOutput(output_name)) {
                        output_shape.clear();
                    }
                    auto output_tensor =
                        ctx.GetOutput(i, output_shape.data(), output_shape.size());
                    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
                    auto shape = tensor_info.GetShape();
                    void *output_buffer = output_tensor.GetTensorMutableRawData();
                    outputs.emplace(output_name,
                    ort_zendnn::OnnxTensorData{
                        ort_zendnn::OnnxTensorInfo{tensor_info.GetElementType(), shape},
                        output_buffer,
                    });
                }
                auto ret = subgraph_primitive->Predict(inputs, outputs);
                num_invocations++;
                if (num_invocations == num_sub_graphs) {
                    num_invocations = 0;
#ifndef _WIN32
                    graph_exe_count++;
#endif
                }
                return ret;
            }
        };

        node_compute_funcs.push_back(std::move(compute_info));
    }
#ifndef _WIN32
    graph_exe_count = 0;
#endif
    return Status::OK();
}

}  // namespace onnxruntime
