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

#include "zendnn_matmul.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "zendnn_util.h"
#include <vector>
#include <unordered_set>
#include <string>

namespace onnxruntime {
namespace ort_zendnn {

ZendnnMatMul::ZendnnMatMul() {}

// This handles ONNX defined "MatMul" as well as two other variations of MatMul
// "MatMulPostOps" is a ZenDNN only fusion of MatMul and upto 32 elementwise or binary ops.
//    See zendnn_subgraph_transformer.cc MatMulBinaryEltwise(...).
// "FusedMatMul" is a ContribOperator defined here:
//    https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FusedMatMul
//    Depending on its attributes "FusedMatMul" can transpose eather input to the MatMul and scale the resulting output
void ZendnnMatMul::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                   ZendnnNode &node) {
    std::unordered_set<std::string> binary_ops = {"Add", "Div", "Mul", "Sub"};
    std::unordered_set<std::string> elementwise_ops = {"Abs", "Elu", "Exp", "LeakyRelu", "Log", "Relu",
                                                       "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"
                                                      };

    auto eng = sp.GetEngine();

    using tag = zendnn::memory::format_tag;
    using dt =  zendnn::memory::data_type;
    bool zendnn_enable_bf16 = false;
    const std::string enable_bf16_env = onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty())
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);

    bool has_postop_fusion = false;
    std::vector<std::string> post_ops;

    if (node.OpType() == "MatMulPostOps") {
        has_postop_fusion = true;
        post_ops = node.GetPostOps();

        int binary_count = 0;
        // Check we have enough inputs for MatMul and the binary post ops
        for (size_t i = 0; i < post_ops.size(); ++i) {
            if (binary_ops.count(post_ops[i]) != 0) {
                assert(node.Input(IN_BINARY_0 + binary_count).Exists());
                binary_count++;
            }
        }
    }

    bool is_fusedmatmul = false;
    bool transA = false;
    bool transBatchA = false;
    bool transB = false;
    bool transBatchB = false;
    float alpha = 1.0;
    if (node.OpType() == "FusedMatMul") {
        // Fused matmul is matmul modified to behave like numpy:
        // https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
        is_fusedmatmul = true;
        transA = GetTransA(node);
        transBatchA = GetTransBatchA(node);
        transB = GetTransB(node);
        transBatchB = GetTransBatchB(node);
        alpha = GetAlpha(node);
    }

    auto src_dims = sp.GetMemory(node.Input(IN_A)).get_desc().dims();
    auto weights_dims = sp.GetMemory(node.Input(IN_B)).get_desc().dims();


    // If this is required for transposed inputs, then this will be done later on in the code.
    if (src_dims.size() != weights_dims.size()) {
        while (src_dims.size() < weights_dims.size() && (!transA && !transBatchA)) {
            src_dims.insert(src_dims.begin(), 1);
        }
        while (src_dims.size() > weights_dims.size() && (!transB && !transBatchB)) {
            weights_dims.insert(weights_dims.begin(), 1);
        }
    }


    auto dataA_dims = src_dims;
    auto ndataA_dims = src_dims.size();
    zendnn::memory::dims transposedA_dims(ndataA_dims, 0);

    auto dataB_dims = weights_dims;
    auto ndataB_dims = weights_dims.size();
    zendnn::memory::dims transposedB_dims(ndataB_dims, 0);

    auto dataA_mem = sp.GetMemory(node.Input(IN_A));
    auto dataB_mem = sp.GetMemory(node.Input(IN_B));


    // Holds transposed matrices A and B. ToDo: Eliminate its usage if in place transpose is possbile for FusedMatmul
    zendnn::memory::desc transposedA_md;
    zendnn::memory transposedA_mem;

    zendnn::memory::desc transposedB_md;
    zendnn::memory transposedB_mem;

    if (is_fusedmatmul) {
        if (transA || transBatchA) {
            zendnn::memory::dims strides = GetStrides(dataA_dims, transA, transBatchA,
                                           transposedA_dims);

            zendnn::memory::desc intermediateA_md;

            if(zendnn_enable_bf16)
                intermediateA_md = zendnn::memory::desc(dataA_dims, dt::bf16, strides);
            else
                intermediateA_md = zendnn::memory::desc(dataA_dims, node.Input(IN_A).Type(), strides);

            zendnn::memory intermediateA_mem = zendnn::memory(intermediateA_md, eng);

            auto traspose_primitive = zendnn::reorder(dataA_mem, intermediateA_mem);
            sp.AddPrimitive(traspose_primitive, {{ZENDNN_ARG_FROM, dataA_mem}, {ZENDNN_ARG_TO, intermediateA_mem}});

            while (transposedA_dims.size() < weights_dims.size()) {
                transposedA_dims.insert(transposedA_dims.begin(), 1);
            }

            // The reorder from above will get the memory in the right order. The next few lines will create a memory and memory descriptor
            // that will have the correct dimentions and correct memory::format
            if(zendnn_enable_bf16)
                transposedA_md = zendnn::memory::desc(transposedA_dims, dt::bf16, sp.GetZendnnFormat(transposedA_dims.size()));
            else
                transposedA_md = zendnn::memory::desc(transposedA_dims, node.Input(IN_A).Type(),
                                                sp.GetZendnnFormat(transposedA_dims.size()));
            transposedA_mem = zendnn::memory(transposedA_md, eng, nullptr);
            void *handle = intermediateA_mem.get_data_handle();
            transposedA_mem.set_data_handle(handle);
        }
        if (transB ||
                transBatchB) {                // Exact same logic for matrix B as used for matrix A
            zendnn::memory::dims strides = GetStrides(dataB_dims, transB, transBatchB,
                                           transposedB_dims);
            zendnn::memory::desc intermediateB_md;
            if(zendnn_enable_bf16)
                intermediateB_md = zendnn::memory::desc(dataB_dims, dt::bf16, strides);
            else
                intermediateB_md = zendnn::memory::desc(dataB_dims, node.Input(IN_B).Type(), strides);
            zendnn::memory intermediateB_mem = zendnn::memory(intermediateB_md, eng);

            auto traspose_primitive = zendnn::reorder(dataB_mem, intermediateB_mem);
            sp.AddPrimitive(traspose_primitive, {{ZENDNN_ARG_FROM, dataB_mem}, {ZENDNN_ARG_TO, intermediateB_mem}});

            while (src_dims.size() > transposedB_dims.size()) {
                transposedB_dims.insert(transposedB_dims.begin(), 1);
            }

            // The reorder from above will get the memory in the right order. The next few lines will create a memory and memory descriptor
            // that will have the correct dimentions and correct memory::format
            if(zendnn_enable_bf16)
                transposedB_md = zendnn::memory::desc(transposedB_dims, dt::bf16, sp.GetZendnnFormat(transposedB_dims.size()));
            else
                transposedB_md = zendnn::memory::desc(transposedB_dims, node.Input(IN_B).Type(),
                                                    sp.GetZendnnFormat(transposedB_dims.size()));
            transposedB_mem = zendnn::memory(transposedB_md, eng, nullptr);
            void *handle = intermediateB_mem.get_data_handle();
            transposedB_mem.set_data_handle(handle);
        }
    }

    zendnn::memory::desc src_md;
    if (transA || transBatchA) {
        src_md = transposedA_md;
    }
    else {
         if(zendnn_enable_bf16)
            src_md = zendnn::memory::desc(src_dims, dt::bf16, tag::any);
        else
            src_md = zendnn::memory::desc(src_dims, node.Input(IN_A).Type(), tag::any);
    }

    zendnn::memory::desc weights_md;
    if (transB || transBatchB) {
        weights_md = transposedB_md;
    }
    else {
        if(zendnn_enable_bf16)
            weights_md = zendnn::memory::desc(weights_dims, dt::bf16, tag::any);
        else
            weights_md = zendnn::memory::desc(weights_dims, node.Input(IN_B).Type(),tag::any);
    }

    auto output_shape = src_dims;
    if (transA || transBatchA) {
        output_shape = transposedA_dims;
    }
    output_shape.pop_back();
    if (transB || transBatchB) {
        output_shape.emplace_back(transposedB_dims.back());
    }
    else {
        output_shape.emplace_back(weights_dims.back());
    }

    for (size_t i = 0; i < output_shape.size() - 2; i++) {
        if (output_shape[i] == 1) {
            if (transB || transBatchB) {
                output_shape[i] = transposedB_dims[i];
            }
            else {
                output_shape[i] = weights_dims[i];
            }
        }
    }

    /*
    create a post op binary with possible unsqueezing in order to make sure zendnn properly broadcast
    current limitation
    1. is no unsqueeze for matmul output as it is not exposed due to post op fusion
    2. the third input has to be reordered to plain format (eg, no memory format propogation if the third input is internal to subgraph)
    3. adding 1s to front (unsqueeze/expand) in logical dims would possibly fail if physcial layout is not plain format
    */
    zendnn::primitive_attr attr;
    if (has_postop_fusion) {
        int binary_count = 0;
        zendnn::post_ops ops;
        for (size_t i = 0; i < post_ops.size(); ++i) {
            zendnn::algorithm algo = zendnn_util::OrtOperatorToZendnnAlgorithm(post_ops[i]);
            // Handle Binary post ops including the input memory
            if (binary_ops.count(post_ops[i]) != 0) {
                auto ori_binary_md = sp.GetMemory(node.Input(IN_BINARY_0 +
                                                  binary_count).Name()).get_desc();
                auto ori_binary_dims = ori_binary_md.dims();
                auto binary_mem_dims = ori_binary_dims;
                if (ori_binary_dims.size() != output_shape.size()) {
                    if (ori_binary_dims.size() > output_shape.size()) {
                        ORT_THROW("add fusion with matmul output broadcasting by unsqueezing is not supported");
                    }
                    // expand the input (from the binary op) if needed to support broadcasting
                    while (binary_mem_dims.size() < output_shape.size()) {
                        binary_mem_dims.insert(binary_mem_dims.begin(), 1);
                    }
                }

                // expand the dims by 1s (should always be possible)
                // will throw exception if not possible
                auto binary_md = ori_binary_md.reshape(binary_mem_dims);
                auto data_type = binary_md.data_type();
                if(zendnn_enable_bf16 && data_type != dt::bf16){
                    binary_md = zendnn::memory::desc({binary_md.dims()}, dt::bf16,
                                                sp.GetZendnnFormat(binary_md.dims().size()));
                }
                // Possible improvment: use format any to choose the best layout
                ops.append_binary(algo, binary_md);
                binary_count++;
                // Handle Elementwise post ops. Some of these require obtaining an 'alpha' attribute
            }
            else if (elementwise_ops.count(post_ops[i]) != 0) {
                float post_op_alpha = 0.0;
                switch (algo) {
                case zendnn::algorithm::eltwise_relu: {
                    // Need to check operator since both Relu and LeakyRelu are covered by algorithm::eltwise_relu
                    if (post_ops[i] == "LeakyRelu") {
                        post_op_alpha = GetFloatAttr(node, "alpha", /*default_alpha*/ 0.01f);
                    }
                    else {
                        post_op_alpha = 0.0;
                    }
                    break;
                }
                case zendnn::algorithm::eltwise_elu: {
                    post_op_alpha = GetFloatAttr(node, "alpha", /*default_alpha*/ 1.0f);
                    break;
                }
                default:
                    post_op_alpha = 0.0;
                }
                ops.append_eltwise(1.0f, algo, post_op_alpha, 0.0f);
            }
        }
        attr.set_post_ops(ops);
    }

    if (is_fusedmatmul) {
        // Set the scaling of output as a post op in the primitive attribute, taking the value from alpha attribute
        std::vector<float> alphaScale({alpha});
        attr.set_output_scales(0, alphaScale);
    }

    auto dst_md = zendnn::memory::desc(output_shape,zendnn_enable_bf16 ? dt::bf16 : node.Output(OUT_Y).Type(),
                                        zendnn::memory::format_tag::any);

    auto matmul_d = zendnn::matmul::desc(src_md, weights_md, dst_md);
    auto matmul_pd = zendnn::matmul::primitive_desc(matmul_d, attr, eng);

    zendnn::memory matmul_src_mem, matmul_weights_mem;
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    zendnn::memory matmul_dst_mem;
    PrimitiveMemInfo mem_info;
    mem_info.ref_count = out_links;
    mem_info.mem_desc  = matmul_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;
    if (mem_info.is_dynamic) {
        matmul_dst_mem = zendnn::memory(matmul_pd.dst_desc(), eng, NULL);
    }
    else {
        matmul_dst_mem = zendnn::memory(matmul_pd.dst_desc(), eng);
    }
    auto matmul_prim = zendnn::matmul(matmul_pd);

    if (transA || transBatchA) {
        matmul_src_mem = transposedA_mem;
    }
    else {
        matmul_src_mem = sp.GetMemoryAndReshape(node.Input(IN_A), matmul_pd.src_desc(),
                                                eng);
    }
    if (transB || transBatchB) {
        matmul_weights_mem = transposedB_mem;
    }
    else {
        matmul_weights_mem = sp.GetMemoryAndReshape(node.Input(IN_B),
                             matmul_pd.weights_desc(), eng);
    }

    // a default memory map for matmul
    std::unordered_map<int, zendnn::memory> mem_map({{ZENDNN_ARG_SRC, matmul_src_mem},
        {ZENDNN_ARG_WEIGHTS, matmul_weights_mem},
        {ZENDNN_ARG_DST, matmul_dst_mem}});

    // add to memory map with extra third input if fused with add
    if (has_postop_fusion) {
        // add to memory map for extra binary inputs
        int binary_count = 0;
        for (size_t i = 0; i < post_ops.size(); ++i) {
            if (binary_ops.count(post_ops[i]) != 0) {
                zendnn::algorithm algo;
                zendnn::memory::desc binary_mem_desc;
                matmul_pd.get_primitive_attr().get_post_ops().get_params_binary(
                    static_cast<int>(i), algo, binary_mem_desc);
                auto binary_post_op_mem = sp.GetMemoryAndReshape(node.Input(
                                              IN_BINARY_0 + binary_count), binary_mem_desc, eng);
                mem_map[ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>
                        (i)) | ZENDNN_ARG_SRC_1] = binary_post_op_mem;
                binary_count++;
            }
        }
    }

    sp.AddPrimitive(matmul_prim, mem_map, mem_info);
    sp.SetMemory(node.Output(OUT_Y), matmul_dst_mem);
}

zendnn::memory::dims ZendnnMatMul::GetStrides(zendnn::memory::dims &data_dims,
        bool trans,
        bool transBatch,
        zendnn::memory::dims &transposed_dims) {
    std::vector<uint32_t> permA;
    std::vector<uint32_t> N_A;
    auto ndata_dims = data_dims.size();
    uint32_t M_A, Batch;
    // Temp vector to hold indices of the dims, will be used to track transposes required
    for (uint32_t i = 0; i < ndata_dims; i++) {
        permA.push_back(i);
    }
    Batch = permA[0];              // Batch Dimension
    M_A = permA[ndata_dims - 1];  // M Dimension
    if (ndata_dims == 4) {        // This will only be used if transBatch is true
        N_A.push_back(permA[ndata_dims - 3]);
    }
    N_A.push_back(permA[ndata_dims - 2]);
    if (trans && !transBatch) {  // Swap last two dimensions for Trans only
        auto n = permA[ndata_dims - 1];
        permA[ndata_dims - 1] = permA[ndata_dims - 2];
        permA[ndata_dims - 2] = n;
    }
    else if (!trans &&
             transBatch) {    // If transBatch only, {Batch, N, M} ---> {N, Batch, M}
        uint32_t i;
        for (i = 0; i < N_A.size(); i++) {
            permA[i] = N_A[i];
        }
        permA[i] = Batch;
    }
    else {    // If both trans and transBatch is true, then end result should be {Batch, N, M} ----> {N, M, Batch}
        uint32_t i;
        for (i = 0; i < N_A.size(); i++) {
            permA[i] = N_A[i];
        }
        permA[i] = M_A;
        permA[i + 1] = Batch;
    }
    zendnn::memory::dims strides(ndata_dims, 0);
    zendnn::memory::dim total_stride = 1;
    for (int i = (int)ndata_dims - 1; i >= 0; i--) {
        transposed_dims[i] = data_dims[permA[i]];
        strides[permA[i]] = total_stride;
        total_stride *= data_dims[permA[i]];
    }

    zendnn::memory::dims strides_inverse;
    strides_inverse.reserve(ndata_dims);
    for (size_t i = 0; i < ndata_dims; ++i) {
        strides_inverse.push_back(strides[ndata_dims - i - 1]);
    }

    return strides;
}

bool ZendnnMatMul::GetTransA(ZendnnNode &node) {
    auto attr = node.Attributes().find("transA");
    if (attr != node.Attributes().end()) {
        return (attr->second().i() != 0);
    }
    return false;
}

bool ZendnnMatMul::GetTransBatchA(ZendnnNode &node) {
    auto attr = node.Attributes().find("transBatchA");
    if (attr != node.Attributes().end()) {
        return (attr->second().i() != 0);
    }
    return false;
}

bool ZendnnMatMul::GetTransB(ZendnnNode &node) {
    auto attr = node.Attributes().find("transB");
    if (attr != node.Attributes().end()) {
        return (attr->second().i() != 0);
    }
    return false;
}

bool ZendnnMatMul::GetTransBatchB(ZendnnNode &node) {
    auto attr = node.Attributes().find("transBatchB");
    if (attr != node.Attributes().end()) {
        return (attr->second().i() != 0);
    }
    return false;
}

float ZendnnMatMul::GetAlpha(ZendnnNode &node) {
    auto attr = node.Attributes().find("alpha");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return 1.0;
}

float ZendnnMatMul::GetFloatAttr(ZendnnNode &node, std::string attr_name,
                                 float default_value) {
    auto attr = node.Attributes().find(attr_name);
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return default_value;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
