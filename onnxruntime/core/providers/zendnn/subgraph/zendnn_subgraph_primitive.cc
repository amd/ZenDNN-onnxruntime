/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

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

#include "zendnn_subgraph_primitive.h"

#include "zendnn_batchnorm.h"
#include "zendnn_binary.h"
#include "zendnn_concat.h"
#include "zendnn_cast.h"
#include "zendnn_conv.h"
#include "zendnn_qconv.h"
#include "zendnn_qbinary.h"
#include "zendnn_dequantizelinear.h"
#include "zendnn_dynamicquantizelinear.h"
#include "zendnn_elementwise.h"
#include "zendnn_flatten.h"
#include "zendnn_gelu.h"
#include "zendnn_gemm.h"
#include "zendnn_inception.h"
#include "zendnn_layernorm.h"
#include "zendnn_lrn.h"
#include "zendnn_matmul.h"
#include "zendnn_matmul_integer.h"
#include "zendnn_pool.h"
#include "zendnn_pow.h"
#include "zendnn_qattention.h"
#include "zendnn_reduce.h"
#include "zendnn_reshape.h"
#include "zendnn_softmax.h"
#include "zendnn_softmaxgrad.h"
#include "zendnn_squeeze.h"
#include "zendnn_sum.h"
#include "zendnn_transpose.h"
#include "zendnn_unsqueeze.h"
#include "zendnn_qpool.h"
#include "zendnn_quantizelinear.h"

#include "zenVitisAI_conv2d.h"
#include "zenVitisAI_pool.h"
#include "zenVitisAI_concatV2.h"

#if defined(ENABLE_TRAINING)
    #include "zendnn_convgrad.h"
    #include "zendnn_poolgrad.h"
    #include "zendnn_relugrad.h"
#endif

#include <inttypes.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

/*
* The ZENDNN_TENSOR_PRINT_MEMORY should always be 0 unless debugging
*
* These macros can be used to print the contents of a ZenDNN tensor
* This can be used to debug and investigate the values inputs and outputs
* of ZenDNN ops.
*
* To use set ZENDNN_TENSOR_PRINT_MEMORY to 1
* Find the operator you want to investigate and add the memory you want to print when
* calling AddPrimitive() for example:
* Change this code:
* ```
* sp.AddPrimitive(elemenwise_primitive,
                  {{ZENDNN_ARG_SRC, elementwise_src_mem}, {ZENDNN_ARG_DST, elementwise_dst_mem}});
* ```
* to
* ```
* sp.AddPrimitive(elemenwise_primitive,
                  {{ZENDNN_ARG_SRC, elementwise_src_mem}, {ZENDNN_ARG_DST, elementwise_dst_mem}},
                  {ZENDNN_ARG_SRC, ZENDNN_ARG_DST});
* ```
* Then rebuild and run the code.
* This is a developer only solution to investigating contents of ZenDNN's tensors.
*/
#define ZENDNN_TENSOR_PRINT_MEMORY 0
#define ZENDNN_TENSOR_PRINT_MEMORY_MAX_TENSOR_ELEMENTS 150

namespace onnxruntime {
namespace ort_zendnn {

template <class Map, class Key>
inline bool Contains(const Map &map, const Key &key) {
    return map.find(key) != map.end();
}

typedef struct zenLibBufState {
    float               *zenLibBufPtr;
    int                 zenLibBufPtrStatus;
} zenLibBufPool;


int zenLibBufPoolSize = 1024;
static zenLibBufPool sZenLibBufPool[1024];

void poolBuf(void *buff, int out_links) {
    for (int i=0; i<zenLibBufPoolSize; i++) {
        if (sZenLibBufPool[i].zenLibBufPtrStatus == 0) {
            sZenLibBufPool[i].zenLibBufPtr = (float *)buff;
            sZenLibBufPool[i].zenLibBufPtrStatus = out_links;
            break;
        }
    }
}

int freeBuf(void *input) {
    int res = -1;
    for (int i=0; i<zenLibBufPoolSize; i++) {
        if (sZenLibBufPool[i].zenLibBufPtr == input) {
            sZenLibBufPool[i].zenLibBufPtrStatus = sZenLibBufPool[i].zenLibBufPtrStatus - 1;
            res = sZenLibBufPool[i].zenLibBufPtrStatus;
            if (res == 0) {
                sZenLibBufPool[i].zenLibBufPtr = NULL;
            }
            break;
        }
    }
    return res;
}


#if ZENDNN_TENSOR_PRINT_MEMORY
void ZendnnSubgraphPrimitive::PrintMemory(const zendnn::memory &mem) {
    auto md = mem.get_desc();
    auto dt = md.data_type();
    auto dims = md.dims();
    if (Product(dims) > ZENDNN_TENSOR_PRINT_MEMORY_MAX_TENSOR_ELEMENTS) {
        printf("tensor too long ignore printing \n");
        return;
    }
    zendnn::memory to_mem;
    if (!IsMemoryInExpectedOrtFormat(md)||
            mem.get_engine().get_kind() != zendnn::engine::kind::cpu) {
        printf("\n print memory reorder started \n");
        zendnn::memory::desc to_md = zendnn::memory::desc(md.dims(), md.data_type(),
                                     GetZendnnFormat(md.dims().size()));
        to_mem = zendnn::memory(to_md, GetCPUEngine());
        auto stream = zendnn::stream(mem.get_engine());
        zendnn::reorder(mem, to_mem).execute(stream, {{ZENDNN_ARG_FROM, mem}, {ZENDNN_ARG_TO, to_mem}});
        stream.wait();
        printf("\n print memory reorder ended \n");
    }
    else {
        to_mem = mem;
    }

    if (dt == zendnn::memory::data_type::f32) {
        std::vector<float> data_vec(Product(dims));
        auto dh = to_mem.get_data_handle();
        for (size_t i = 0; i < to_mem.get_desc().get_size(); ++i) {
            ((char *)data_vec.data())[i] = ((char *)dh)[i];
        }

        std::cout << "[";
        for (auto &data : data_vec) {
            std::cout << std::setprecision(6) << data;
            if (&data != &data_vec.back()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    else if (dt == zendnn::memory::data_type::u8) {
        std::vector<uint8_t> data_vec(Product(dims));
        auto dh = to_mem.get_data_handle();
        for (size_t i = 0; i < to_mem.get_desc().get_size(); ++i) {
            ((char *)data_vec.data())[i] = ((char *)dh)[i];
        }

        std::cout << "[";
        for (auto &data : data_vec) {
            std::cout << +data;
            if (&data != &data_vec.back()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    else if (dt == zendnn::memory::data_type::s8) {
        std::vector<int8_t> data_vec(Product(dims));
        auto dh = to_mem.get_data_handle();
        for (size_t i = 0; i < to_mem.get_desc().get_size(); ++i) {
            ((char *)data_vec.data())[i] = ((char *)dh)[i];
        }

        std::cout << "[";
        for (auto &data : data_vec) {
            std::cout << +data;
            if (&data != &data_vec.back()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    else if (dt == zendnn::memory::data_type::s32) {
        std::vector<int32_t> data_vec(Product(dims));
        auto dh = to_mem.get_data_handle();
        for (size_t i = 0; i < to_mem.get_desc().get_size(); ++i) {
            ((char *)data_vec.data())[i] = ((char *)dh)[i];
        }

        std::cout << "[";
        for (auto &data : data_vec) {
            std::cout << data;
            if (&data != &data_vec.back()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    else {
        ORT_THROW("Cannot print such data type");
    }
}
#endif // ZENDNN_TENSOR_PRINT_MEMORY

int Product(zendnn::memory::dims d) {
    int result = 1;
    for (const auto &e : d) {
        result *= (int)e;
    }
    return result;
}

void ZendnnSubgraphPrimitive::AddKernels() {
    std::unordered_set<std::string> binary_ops = {"Add", "Div", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Mul", "Sub"};
    std::unordered_set<std::string> qbinary_ops = {"QLinearAdd", /*, "QLinearMul", "QLinearSub"*/};
    std::unordered_set<std::string> elementwise_ops = {"Abs", "Clip", "Elu", "Exp", "LeakyRelu", "Log", "Relu", "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"};
    std::unordered_set<std::string> pool_ops = {"AveragePool", "GlobalAveragePool", "GlobalMaxPool", "MaxPool"};
    std::unordered_set<std::string> vitisAI_pool_ops = {"VitisAIMaxPool", "VitisAIAvgPool"};
    std::unordered_set<std::string> vitisAI_conv_ops = {"VitisAIConv2D", "VitisAIConv2DWithSum", "VitisAIDepthwiseConv2D", "VitisAIConv2DWithoutBias"};
    std::unordered_set<std::string> reduce_ops = {"ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare"};

    auto indices = subgraph_->GetZendnnNodesInTopologicalOrder();
    for (auto index : indices) {
        auto &node = *(subgraph_->GetZendnnNode(index));
        if (node.OpType() == "BatchNormalization" || node.OpType() == "BatchnormRelu") {
            ZendnnBatchNorm().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Concat") {
            ZendnnConcat().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "zenInceptionIPC") {
            ZendnnInception().CreatePrimitive(*this, node);
        }
        else if (binary_ops.count(node.OpType())) {
            ZendnnBinary().CreatePrimitive(*this, node);
        }
        else if (qbinary_ops.count(node.OpType())) {
            ZendnnQBinary().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Cast") {
            ZendnnCast().CreatePrimitive(*this, node);
        }
        else if (vitisAI_conv_ops.count(node.OpType())) {
            ZenVitisAIConv2D().CreatePrimitive(*this, node);
        }
        else if (vitisAI_pool_ops.count(node.OpType())) {
            ZenVitisAIPool().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "VitisAIConcatV2") {
            ZenVitisAIConcat().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "QLinearConv" ||
                 node.OpType() == "QLinearConv_v1" ||
                 node.OpType() == "QLinearConv_v2" ||
                 node.OpType() == "QConvAdd" ||
                 node.OpType() == "QConvAdd_v1" ||
                 node.OpType() == "QConvRelu" ||
                 node.OpType() == "QConvAddRelu") {
            ZendnnQConv().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Conv" ||
                 node.OpType() == "ConvRelu" ||
                 node.OpType() == "ConvClip" ||
                 node.OpType() == "ConvElu" ||
                 node.OpType() == "ConvSwish" ||
                 node.OpType() == "ConvAdd" ||
                 node.OpType() == "ConvAddRelu") {
            if (!node.isInplaceMemoryNode) {
                ZendnnConv().CreatePrimitive(*this, node);
            }
        }
        else if (node.OpType() == "DequantizeLinear") {
            ZendnnDequantizeLinear().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "QLinearGlobalAveragePool") {
            ZendnnQPool().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "QuantizeLinear") {
            ZendnnQuantizeLinear().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "DynamicQuantizeLinear") {
            ZendnnDynamicQuantizeLinear().CreatePrimitive(*this, node);
        }
        else if (elementwise_ops.count(node.OpType())) {
            ZendnnElementwise().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "FastGelu") {
            ZendnnGelu().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Flatten") {
            ZendnnFlatten().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Gelu" || node.OpType() == "BiasGelu") {
            ZendnnGelu().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "LayerNormalization" ||
                 node.OpType() == "SkipLayerNormalization") {
            ZendnnLayerNorm().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Gemm") {
            ZendnnGemm().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "LRN") {
            ZendnnLrn().CreatePrimitive(*this, node);
            // MatMulPostOps is a ZenDNN only fusion of MatMul and upto 32 elementwise or binary ops
            // FusedMatMul is a ContribOperator defined here:
            //    https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMul
        }
        else if (node.OpType() == "MatMul" || node.OpType() == "MatMulPostOps" ||
                 node.OpType() == "FusedMatMul") {
            ZendnnMatMul().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "MatMulInteger" ||
                 node.OpType() == "MatMulIntegerPostOps") {
            ZendnnMatMulInteger().CreatePrimitive(*this, node);
        }
        else if (pool_ops.count(node.OpType())) {
            ZendnnPool().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Pow") {
            ZendnnPow().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "QAttention") {
            ZendnnQAttention().CreatePrimitive(*this, node);
        }
        else if (reduce_ops.count(node.OpType())) {
            ZendnnReduce().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Reshape") {
            ZendnnReshape().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Softmax") {
            ZendnnSoftmax().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Squeeze") {
            ZendnnSqueeze().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Sum") {
            ZendnnSum().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Transpose") {
            ZendnnTranspose().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "Unsqueeze") {
            ZendnnUnsqueeze().CreatePrimitive(*this, node);
#if defined(ENABLE_TRAINING)
        }
        else if (node.OpType() == "AveragePoolGrad" || node.OpType() == "MaxPoolGrad") {
            ZendnnPoolGrad().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "ConvGrad") {
            ZendnnConvGrad().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "ReluGrad") {
            ZendnnReluGrad().CreatePrimitive(*this, node);
        }
        else if (node.OpType() == "SoftmaxGrad") {
            ZendnnSoftmaxGrad().CreatePrimitive(*this, node);
#endif
        }
        else {
            throw std::invalid_argument("Kernel not found");
        }
    }
}

ZendnnSubgraphPrimitive::ZendnnSubgraphPrimitive(ort_zendnn::ZendnnSubgraph
        &zendnn_subgraph, AllocatorPtr allocator) {
    subgraph_ = &zendnn_subgraph;
    if (zendnn_engine_get_count(zendnn_engine_kind_t::zendnn_cpu)) {
        cpu_engine_ = zendnn::engine(zendnn::engine::kind::cpu, 0);
    }

    if (zendnn_engine_get_count(zendnn_engine_kind_t::zendnn_gpu)) {
        gpu_engine_ = zendnn::engine(zendnn::engine::kind::gpu, 0);
    }

    alloc_ptr = allocator;
    const std::string cpu_mem_alloc_env =
        onnxruntime::GetEnvironmentVar("ONNXRT_ZENDNN_CPU_ALLOC");
    if (!cpu_mem_alloc_env.empty()) {
        cpu_mem_alloc_ = (std::stoi(cpu_mem_alloc_env) == 0 ? false : true);
    }
}

bool ZendnnSubgraphPrimitive::UseCPUAllocator() {
    return cpu_mem_alloc_;
}

bool ZendnnSubgraphPrimitive::IsDynamic() {
    return subgraph_->IsDynamic();
}

bool ZendnnSubgraphPrimitive::IsScalar(const ZendnnTensor &tensor) {
    return Contains(input_is_scalar_, tensor.Name());
}

void ZendnnSubgraphPrimitive::Compile(const
                                      std::unordered_map<std::string, OnnxTensorData> &inputs) {
    //if already compiled once and is not dynamic, then don't compile again
    if (!shape_key_.empty() && !IsDynamic()) {
        return;
    }

    std::string key;
    for (auto input : inputs) {
        for (auto dim : input.second.tensor_info.shape) {
            std::ostringstream o;
            o << dim;
            key += o.str();
            key += ",";
        }
        key += "|";
    }
    // if key different from shape key, update and recompile
    if (key != shape_key_) {
        shape_key_ = key;
    }
    else {
        return;
    }
    if (IsDynamic()) {
        LOGS_DEFAULT(INFO) << "Dynamic Compile";
    }
    else {
        LOGS_DEFAULT(INFO) << "Static Compile";
    }

    inputs_.clear();
    intermediates_.clear();
    outputs_.clear();
    outputs_are_always_copied_.clear();
    inputs_md_.clear();
    outputs_md_.clear();
    net_.clear();
    net_args_.clear();
    reshapes_.clear();
    scalar_outputs_.clear();
    //initializer should not be cleared upon recompile
    //initializers_.clear();

    for (auto nodearg : subgraph_->GetZendnnInputs()) {
        auto zendnn_tensor_name = nodearg->Name();
        auto zendnn_data_type = nodearg->Type();
        zendnn::memory::dims zendnn_dims = inputs.at(
                                               zendnn_tensor_name).tensor_info.shape;
        if (zendnn_dims.size() == 0) {
            zendnn_dims.push_back(1);
            input_is_scalar_.insert(zendnn_tensor_name);
        }
        auto zendnn_format = GetZendnnFormat(zendnn_dims.size());
        auto input_md = zendnn::memory::desc(zendnn_dims, zendnn_data_type,
                                             zendnn_format);
        inputs_md_.emplace(zendnn_tensor_name, input_md);
        auto engine = GetCPUEngine();
        auto input_mem = zendnn::memory(input_md, engine,
                                        inputs.at(zendnn_tensor_name).buffer);
        inputs_.emplace(zendnn_tensor_name, input_mem);
    }

    AddInitializers();
    AddKernels();
    AddOutputs();
}

zendnn::memory::format_tag ZendnnSubgraphPrimitive::GetZendnnFormat(
    size_t dim_size) {
    zendnn::memory::format_tag source_format = zendnn::memory::format_tag::any;
    switch (dim_size) {
    case 1: {
        source_format = zendnn::memory::format_tag::x;
        break;
    }
    case 2: {
        source_format = zendnn::memory::format_tag::nc;
        break;
    }
    case 3: {
        source_format = zendnn::memory::format_tag::ncw;
        break;
    }
    case 4: {
        source_format = zendnn::memory::format_tag::nchw;
        break;
    }
    case 5: {
        source_format = zendnn::memory::format_tag::ncdhw;
        break;
    }
    case 6: {
        source_format = zendnn::memory::format_tag::abcdef;
        break;
    }
    case 7: {
        source_format = zendnn::memory::format_tag::abcdefg;
        break;
    }
    case 8: {
        source_format = zendnn::memory::format_tag::abcdefgh;
        break;
    }
    case 9: {
        source_format = zendnn::memory::format_tag::abcdefghi;
        break;
    }
    case 10: {
        source_format = zendnn::memory::format_tag::abcdefghij;
        break;
    }
    case 11: {
        source_format = zendnn::memory::format_tag::abcdefghijk;
        break;
    }
    case 12: {
        source_format = zendnn::memory::format_tag::abcdefghijkl;
        break;
    }
    default: {
        source_format = zendnn::memory::format_tag::any;
        break;
    }
    }
    return source_format;
}

zendnn::engine ZendnnSubgraphPrimitive::GetCPUEngine() {
    return cpu_engine_;
}

zendnn::engine ZendnnSubgraphPrimitive::GetEngine() {
    if (gpu_engine_) {
        return gpu_engine_;
    }
    return cpu_engine_;
}

zendnn::stream ZendnnSubgraphPrimitive::GetStream() {
    return zendnn::stream(GetEngine());
}

void ZendnnSubgraphPrimitive::AddInitializers() {
    for (auto nodearg : subgraph_->GetZendnnInitializers()) {
        auto zendnn_tensor_name = nodearg->Name();
        if (!Contains(initializers_, zendnn_tensor_name)) {
            initializers_.insert(std::pair<std::string, std::vector<zendnn::memory> >
                                 (zendnn_tensor_name, std::vector<zendnn::memory>()));
        }
    }
}

void ZendnnSubgraphPrimitive::AddOutputs() {
    for (auto tensor : subgraph_->GetZendnnOutputs()) {
        auto zendnn_data_type = tensor->Type();
        auto zendnn_tensor_name = tensor->Name();
        auto engine = GetCPUEngine();
        auto output_mem_zendnn = GetMemory(zendnn_tensor_name);
        auto output_md = zendnn::memory::desc(output_mem_zendnn.get_desc().dims(),
                                              zendnn_data_type, GetZendnnFormat(output_mem_zendnn.get_desc().dims().size()));
        // if output already in correct memory format, just place it to outputs instead of reorder
        bool copy_output = outputs_are_always_copied_.find(zendnn_tensor_name) !=
                           outputs_are_always_copied_.end();
        if (output_mem_zendnn.get_desc() == output_md &&
                output_mem_zendnn.get_engine() == engine && !copy_output) {
            outputs_.emplace(zendnn_tensor_name, output_mem_zendnn);
        }
        else {
            auto output_mem = zendnn::memory(output_md, engine, nullptr);
            AddPrimitive(zendnn::reorder(output_mem_zendnn, output_mem), {{ZENDNN_ARG_FROM, output_mem_zendnn},
                {ZENDNN_ARG_TO, output_mem}
            });
            outputs_.emplace(zendnn_tensor_name, output_mem);
        }
    }
}

bool ZendnnSubgraphPrimitive::HasMemory(std::string memory_name,
                                        zendnn::memory::desc mem_desc, zendnn::engine eng) {
    if (Contains(initializers_, memory_name)) {
        for (auto &mem : initializers_.at(memory_name)) {
            if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
                return true;
            }
        }
    }

    if (Contains(inputs_, memory_name)) {
        auto &mem = inputs_.at(memory_name);
        if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
            return true;
        }
    }

    if (Contains(intermediates_, memory_name)) {
        for (auto &mem : intermediates_.at(memory_name)) {
            if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
                return true;
            }
        }
    }

    return false;
}

void ZendnnSubgraphPrimitive::SetMemory(const ZendnnTensor &tensor,
                                        zendnn::memory mem, bool always_copy_output, bool is_scalar) {
    if (always_copy_output) {
        outputs_are_always_copied_.insert(tensor.Name());
    }
    if (is_scalar) {
        // output may be input for another subgraph node
        input_is_scalar_.insert(tensor.Name());
        scalar_outputs_.insert(tensor.Name());
    }
    SetMemory(tensor.Name(), mem);
}

zendnn::memory ZendnnSubgraphPrimitive::GetMemory(const ZendnnTensor &tensor) {
    std::string memory_name = tensor.Name();
    if (Contains(initializers_, memory_name)) {
        if (!initializers_.at(memory_name).empty()) {
            return initializers_.at(memory_name)[0];
        }
    }

    if (Contains(inputs_, memory_name)) {
        return inputs_.at(memory_name);
    }

    if (Contains(intermediates_, memory_name)) {
        return intermediates_.at(memory_name)[0];
    }

    throw std::invalid_argument("cannot find memory");
}

zendnn::memory ZendnnSubgraphPrimitive::GetMemory(const ZendnnTensor &tensor,
        zendnn::memory::desc mem_desc, zendnn::engine eng) {
    std::string memory_name = tensor.Name();
    if (Contains(initializers_, memory_name)) {
        for (auto &mem : initializers_.at(memory_name)) {
            if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
                return mem;
            }
        }
    }

    if (Contains(inputs_, memory_name)) {
        auto &mem = inputs_.at(memory_name);
        if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
            return mem;
        }
    }

    if (Contains(intermediates_, memory_name)) {
        for (auto &mem : intermediates_.at(memory_name)) {
            if (mem.get_engine() == eng && mem.get_desc() == mem_desc) {
                return mem;
            }
        }
    }

    throw std::invalid_argument("cannot find memory");
}

void ZendnnSubgraphPrimitive::SetMemory(std::string memory_name,
                                        zendnn::memory mem) {
    if (Contains(intermediates_, memory_name)) {
        for (auto &tmp_mem : intermediates_.at(memory_name)) {
            if (tmp_mem == mem) {
                throw std::invalid_argument("setting duplicate memory");
            }
        }
        intermediates_.at(memory_name).push_back(mem);
    }
    else {
        intermediates_.insert(std::pair<std::string, std::vector<zendnn::memory> >
                              (memory_name, std::vector<zendnn::memory>()));
        intermediates_[memory_name].push_back(mem);
    }
}

void ZendnnSubgraphPrimitive::SetInitializer(std::string memory_name,
        zendnn::memory mem) {
    if (Contains(initializers_, memory_name)) {
        for (auto &tmp_mem : initializers_.at(memory_name)) {
            if (tmp_mem == mem) {
                throw std::invalid_argument("setting duplicate initializer");
            }
        }
        initializers_.at(memory_name).push_back(mem);
    }
    else {
        initializers_.insert(std::pair<std::string, std::vector<zendnn::memory> >
                             (memory_name, std::vector<zendnn::memory>()));
        initializers_[memory_name].push_back(mem);
    }
}



zendnn::memory ZendnnSubgraphPrimitive::GetMemoryAndReshape(
    const ZendnnTensor &tensor, zendnn::memory::desc mem_desc, zendnn::engine eng,
    bool transpose) {
    // if found just return
    if (HasMemory(tensor.Name(), mem_desc, eng)) {
        return GetMemory(tensor, mem_desc, eng);
    }

    // is non overridable constant initializer (assume already in memory (runtime))
    bool is_constant = Contains(initializers_, tensor.Name());
    if (is_constant) {
        LOGS_DEFAULT(INFO) << "initializer cache started";
    }
    // will get the first memory with matching name
    auto mem_from = GetMemory(tensor);
    auto mem_to = zendnn::memory(mem_desc, eng, NULL);

    // if it is a reshape, ensure reorder is possible by making the same dims
    if (mem_from.get_desc().dims() != mem_to.get_desc().dims() || transpose) {
        auto mem_from_dims = mem_from.get_desc().dims();
        auto mem_to_dims = mem_to.get_desc().dims();
        if (Product(mem_from_dims) != Product(mem_to_dims)) {
            LOGS_DEFAULT(ERROR) << mem_from_dims;
            LOGS_DEFAULT(ERROR) << mem_to_dims;
            throw std::invalid_argument("not a valid reshape, inconsistent dim product");
        }
        //keep the same data type from mem_from but reshape the dims with mem_desc
        auto mem_from_reshape_md = mem_from.get_desc();
        if (transpose) {
            //hard coded to transpose 2 dimensional matrix
            //TODO: expand to arbitrary permutation or transpose on given 2 dims for higher dimensional tensors
            mem_from_reshape_md = mem_from_reshape_md.permute_axes({1, 0});
        }
        mem_from_reshape_md = mem_from_reshape_md.reshape(mem_desc.dims());
        auto mem_from_reshape = zendnn::memory(mem_from_reshape_md,
                                               mem_from.get_engine(), nullptr);
        if (is_constant) {  // if constant, do reshape now
            LOGS_DEFAULT(INFO) << "reshaped now";
            //use the stream as a hint to make sure data handle gets set
            zendnn::stream s{eng};
            mem_from_reshape.set_data_handle(mem_from.get_data_handle(),s);
            s.wait();
        }
        else {
            AddReshape(mem_from, mem_from_reshape);
        }
        if (mem_from_reshape.get_desc() == mem_to.get_desc() &&
                mem_from_reshape.get_engine() == mem_to.get_engine()) {
            mem_to = mem_from_reshape;
        }
        else {                // after reshape still need to reorder
            if (is_constant) {  // execute reorder now if constant
                mem_to = zendnn::memory(mem_desc, eng);
                zendnn::stream s{eng};
                zendnn::reorder(mem_from_reshape, mem_to).execute(s, mem_from_reshape, mem_to);
                s.wait();
            }
            else {
                PrimitiveMemInfo mem_info;
                mem_info.is_dynamic = UseCPUAllocator();
                mem_info.ref_count = 1;
                if (mem_info.is_dynamic == false) {
                    mem_to = zendnn::memory(mem_desc, eng);
                }
                mem_info.mem_desc  = mem_to.get_desc();
                AddPrimitive(zendnn::reorder(mem_from_reshape, mem_to), {{ZENDNN_ARG_FROM, mem_from_reshape},
                    {ZENDNN_ARG_TO, mem_to}
                }, mem_info);
            }
        }
    }
    else {                // same shape, save to reorder
        if (is_constant) {  // execute reorder now if constant
            mem_to = zendnn::memory(mem_desc, eng);
            zendnn::stream s{eng};
            zendnn::reorder(mem_from, mem_to).execute(s, mem_from, mem_to);
            s.wait();
        }
        else {
            PrimitiveMemInfo mem_info;
            mem_info.is_dynamic = UseCPUAllocator();
            mem_info.ref_count = 1;
            if (mem_info.is_dynamic == false) {
                mem_to = zendnn::memory(mem_desc, eng);
            }
            mem_info.mem_desc  = mem_to.get_desc();
            AddPrimitive(zendnn::reorder(mem_from, mem_to), {{ZENDNN_ARG_FROM, mem_from}, {ZENDNN_ARG_TO, mem_to}},
            mem_info);
        }
    }

    if (is_constant) {  // initializer should stay even after dynamic recompile
        SetInitializer(tensor.Name(), mem_to);
    }
    return mem_to;
}

/* This API is part of reorder optimization trick. s8(-128, 127)->relu = u8(0, 127).
 * Please use this with caution. Currently it's usage is limited to selected GetMemory() cases.
*/
zendnn::memory ZendnnSubgraphPrimitive::GetMemoryAndReshapeByHandle(
    const ZendnnTensor &tensor, zendnn::memory::desc mem_desc, zendnn::engine eng,
    bool transpose) {
    // if found just return
    if (HasMemory(tensor.Name(), mem_desc, eng)) {
        return GetMemory(tensor, mem_desc, eng);
    }

    // is non overridable constant initializer (assume already in memory (runtime))
    bool is_constant = Contains(initializers_, tensor.Name());
    if (is_constant) {
        LOGS_DEFAULT(INFO) << "initializer cache started";
    }
    // will get the first memory with matching name
    auto mem_from = GetMemory(tensor);
    auto mem_to = zendnn::memory(mem_desc, eng);

    // if it is a reshape, ensure reorder is possible by making the same dims
    if (mem_from.get_desc().dims() != mem_to.get_desc().dims() || transpose) {
        auto mem_from_dims = mem_from.get_desc().dims();
        auto mem_to_dims = mem_to.get_desc().dims();
        if (Product(mem_from_dims) != Product(mem_to_dims)) {
            LOGS_DEFAULT(ERROR) << mem_from_dims;
            LOGS_DEFAULT(ERROR) << mem_to_dims;
            throw std::invalid_argument("not a valid reshape, inconsistent dim product");
        }
        //keep the same data type from mem_from but reshape the dims with mem_desc
        auto mem_from_reshape_md = mem_from.get_desc();
        if (transpose) {
            //hard coded to transpose 2 dimensional matrix
            //TODO: expand to arbitrary permutation or transpose on given 2 dims for higher dimensional tensors
            mem_from_reshape_md = mem_from_reshape_md.permute_axes({1, 0});
        }
        mem_from_reshape_md = mem_from_reshape_md.reshape(mem_desc.dims());
        auto mem_from_reshape = zendnn::memory(mem_from_reshape_md,
                                               mem_from.get_engine(), nullptr);
        if (is_constant) {  // if constant, do reshape now
            LOGS_DEFAULT(INFO) << "reshaped now";
            //use the stream as a hint to make sure data handle gets set
            zendnn::stream s{eng};
            mem_from_reshape.set_data_handle(mem_from.get_data_handle(),s);
            s.wait();
        }
        else {
            AddReshape(mem_from, mem_from_reshape);
        }
        if (mem_from_reshape.get_desc() == mem_to.get_desc() &&
                mem_from_reshape.get_engine() == mem_to.get_engine()) {
            mem_to = mem_from_reshape;
        }
        else {                // after reshape still need to reorder
            if (is_constant) {  // execute reorder now if constant
                zendnn::stream s{eng};
                zendnn::reorder(mem_from_reshape, mem_to).execute(s, mem_from_reshape, mem_to);
                s.wait();
            }
            else {
                AddPrimitive(zendnn::reorder(mem_from_reshape, mem_to), {{ZENDNN_ARG_FROM, mem_from_reshape},
                    {ZENDNN_ARG_TO, mem_to}
                });
            }
        }
    }
    else {                // same shape, save to reorder
        if (is_constant) {  // execute reorder now if constant
            zendnn::stream s{eng};
            zendnn::reorder(mem_from, mem_to).execute(s, mem_from, mem_to);
            s.wait();
        }
        else {
            if (mem_from.get_desc().dims() == mem_to.get_desc().dims() &&
                    mem_from.get_desc().data.format_kind == mem_to.get_desc().data.format_kind &&
                    (mem_from.get_desc().data_type() == zendnn::memory::data_type::s8 &&
                     mem_to.get_desc().data_type() == zendnn::memory::data_type::u8)) {
                mem_to.set_data_handle(mem_from.get_data_handle());
            }
            else {
                AddPrimitive(zendnn::reorder(mem_from, mem_to), {{ZENDNN_ARG_FROM, mem_from}, {ZENDNN_ARG_TO, mem_to}});
            }
        }
    }

    if (is_constant) {  // initializer should stay even after dynamic recompile
        SetInitializer(tensor.Name(), mem_to);
    }
    return mem_to;
}

zendnn::memory ZendnnSubgraphPrimitive::GetMemoryInOrtFormat(
    const ZendnnTensor &tensor, const zendnn::engine &eng) {
    auto from_mem = GetMemory(tensor);
    auto from_desc = from_mem.get_desc();
    auto from_dims = from_desc.dims();
    if (!IsMemoryInExpectedOrtFormat(from_desc)) {
        zendnn::memory::desc to_md = zendnn::memory::desc(from_dims, tensor.Type(),
                                     GetZendnnFormat(from_dims.size()));
        PrimitiveMemInfo mem_info;
        mem_info.is_dynamic = UseCPUAllocator();
        mem_info.ref_count = 1;
        mem_info.mem_desc  = to_md;
        mem_info.variable_inputs = 0;

        zendnn::memory to_mem;
        if (mem_info.is_dynamic == true) {
            to_mem = zendnn::memory(to_md, eng, NULL);
        }
        else {
            to_mem = zendnn::memory(to_md, eng);
        }
        AddPrimitive(zendnn::reorder(from_mem, to_mem), {{ZENDNN_ARG_FROM, from_mem},
            {ZENDNN_ARG_TO, to_mem}
        }, mem_info);
        return to_mem;
    }
    else {
        // If using GPU this will move the memory from the CPU to the GPU.
        return GetMemoryAndReshape(tensor, from_desc, eng);
    }
}

bool ZendnnSubgraphPrimitive::IsMemoryInExpectedOrtFormat(
    const zendnn::memory::desc &desc) const {
    if (desc.data.format_kind != zendnn_blocked) {
        return false;
    }
    if (desc.data.format_desc.blocking.inner_nblks != 0) {
        return false;
    }
    auto strides = desc.data.format_desc.blocking.strides;
    // if a data format is zendnn_format::abcd... the stride will go from largest to smallest
    // if for example we have a shape {2,3,4} we expect a stride of {12, 4, 1} if it were
    // of zendnn_format::abc if instead the stride were {12, 1, 4} that would be zendnn_format::acb
    // which does not match what is expected from Onnxruntime.
    for (size_t i = 1; i < desc.dims().size(); ++i) {
        if (strides[i - 1] < strides[i]) {
            return false;
        }
    }
    return true;
}

void ZendnnSubgraphPrimitive::IncMemoryRefCount(zendnn::memory &mem) {

    size_t index = 0, res = 0;
    //Search all the primitives for the dest memory
    for (index = 0; index < net_.size(); ++index) {
        std::unordered_map<int, zendnn::memory> mem_map = net_args_.at(index);
        if (mem == mem_map[ZENDNN_ARG_DST]) {
            res = index;
            break;
        }
    }
    if (index == net_.size()) {
        return;
    }

    PrimitiveMemInfo mem_info  = net_prim_mem_info_.at(res);
    mem_info.ref_count = mem_info.ref_count + 1;
    net_prim_mem_info_[res] = mem_info;
    return;
}

void ZendnnSubgraphPrimitive::AddReshape(zendnn::memory src,
        zendnn::memory dst) {
    LOGS_DEFAULT(INFO) << "reshape queued";
    reshapes_.push_back({src, dst});
}

void ZendnnSubgraphPrimitive::AddPrimitive(zendnn::primitive prim,
        std::unordered_map<int, zendnn::memory> mem_map,
        PrimitiveMemInfo primitive_mem_info,
        std::vector<int> items_to_print) {
    net_.push_back(prim);
    net_args_.push_back(mem_map);
    net_prim_mem_info_.push_back(primitive_mem_info);
    for (auto e : items_to_print) {
        items_to_print_.push_back({int(net_.size() - 1), e});
    }
}

onnxruntime::common::Status ZendnnSubgraphPrimitive::Predict(
    const std::unordered_map<std::string, OnnxTensorData> &inputs,
    const std::unordered_map<std::string, OnnxTensorData> &outputs) {

    auto stream = GetStream();

    for (auto &input : inputs) {
        if (Contains(inputs_, input.first)) {
            inputs_.at(input.first).set_data_handle(input.second.buffer, stream);
            stream.wait();
        }
    }

    for (auto &output : outputs) {
        if (Contains(outputs_, output.first)) {
            outputs_.at(output.first).set_data_handle(output.second.buffer, stream);
            stream.wait();
        }
    }

    // reshapes (eg, unsqueeze)
    // it is safe to set data handle because all external data handles have been set and zendnn managed memory data handles will not change
    for (auto &reshape_pair : reshapes_) {
        reshape_pair.second.set_data_handle(reshape_pair.first.get_data_handle(),
                                            stream);
        stream.wait();
    }


    for (size_t i = 0; i < net_.size(); ++i) {
        // Get the memory pool buffer usage
        PrimitiveMemInfo mem_info  = net_prim_mem_info_.at(i);
        std::unordered_map<int, zendnn::memory> mem_map = net_args_.at(i);
        //Get the buffer from the memory pool
        if (mem_info.is_dynamic == true) {
            void *buff = alloc_ptr->Alloc(mem_info.mem_desc.get_size());
            poolBuf(buff, mem_info.ref_count);
            // Check if there is any reshape required on this memory
            for (auto &reshape_pair : reshapes_) {
                if (reshape_pair.first == mem_map[ZENDNN_ARG_DST]) {
                    mem_map[ZENDNN_ARG_DST].set_data_handle(buff);
                    reshape_pair.second.set_data_handle(reshape_pair.first.get_data_handle(),
                                                        stream);
                    stream.wait();
                    break;
                }
            }
            mem_map[ZENDNN_ARG_DST].set_data_handle(buff);
        }
        net_.at(i).execute(stream, net_args_.at(i));
        stream.wait();
        if (mem_info.variable_inputs <=0) {
            if (mem_map.count(ZENDNN_ARG_SRC_0) != 0) {
                if (freeBuf(mem_map[ZENDNN_ARG_SRC_0].get_data_handle()) == 0) {
                    alloc_ptr->Free(mem_map[ZENDNN_ARG_SRC_0].get_data_handle());
                }
            }
            if (mem_map.count(ZENDNN_ARG_SRC_1) != 0) {
                if (freeBuf(mem_map[ZENDNN_ARG_SRC_1].get_data_handle()) == 0) {
                    alloc_ptr->Free(mem_map[ZENDNN_ARG_SRC_1].get_data_handle());
                }
            }
            if (mem_map.count(ZENDNN_ARG_SRC_2) != 0) {
                if (freeBuf(mem_map[ZENDNN_ARG_SRC_2].get_data_handle()) == 0) {
                    alloc_ptr->Free(mem_map[ZENDNN_ARG_SRC_2].get_data_handle());
                }
            }
            if (mem_map.count(ZENDNN_ARG_SRC_3) != 0) {
                if (freeBuf(mem_map[ZENDNN_ARG_SRC_3].get_data_handle()) == 0) {
                    alloc_ptr->Free(mem_map[ZENDNN_ARG_SRC_3].get_data_handle());
                }
            }
        }
        else {
            for (int j = 0; j < mem_info.variable_inputs; j++) {
                if (mem_map.count(ZENDNN_ARG_MULTIPLE_SRC + j) != 0) {
                    if (freeBuf(mem_map[ZENDNN_ARG_MULTIPLE_SRC + j].get_data_handle()) == 0) {
                        alloc_ptr->Free(mem_map[ZENDNN_ARG_MULTIPLE_SRC + j].get_data_handle());
                    }
                }
            }
        }
#if ZENDNN_TENSOR_PRINT_MEMORY
        //for debug memory purpose
        for (auto e : items_to_print_) {
            auto net_index = e.first;
            auto net_arg_index = e.second;
            if (net_index == static_cast<int>(i)) {
                PrintMemory(net_args_.at(i)[net_arg_index]);
            }
        }
#endif  //ZENDNN_TENSOR_PRINT_MEMORY
    }
    //resetMemPool();
    for (int i=0; i<zenLibBufPoolSize; i++) {
        if (sZenLibBufPool[i].zenLibBufPtrStatus > 0) {
            alloc_ptr->Free(sZenLibBufPool[i].zenLibBufPtr);
            sZenLibBufPool[i].zenLibBufPtrStatus=0;
            sZenLibBufPool[i].zenLibBufPtr = NULL;
        }
    }

    return Status::OK();
}

bool ZendnnSubgraphPrimitive::IsScalarOutput(const std::string &name) {
    return Contains(scalar_outputs_,name);
}

zendnn::memory::desc ZendnnSubgraphPrimitive::GetOutputInfo(std::string name) {
    if (Contains(outputs_, name)) {
        return outputs_.at(name).get_desc();
    }
    throw std::invalid_argument("no such output exists");
}

void ZendnnSubgraphPrimitive::SetOrderedInputs(std::vector<std::string>
        &&inputs) {
    ordered_inputs_ = std::move(inputs);
}

void ZendnnSubgraphPrimitive::SetOrderedOutputs(std::vector<std::string>
        &&outputs) {
    ordered_outputs_ = std::move(outputs);
}

const std::vector<std::string> &ZendnnSubgraphPrimitive::GetOrderedInputs()
const {
    return ordered_inputs_;
}

const std::vector<std::string> &ZendnnSubgraphPrimitive::GetOrderedOutputs()
const {
    return ordered_outputs_;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
