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

#include "zendnn_gemm.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnGemm::ZendnnGemm() {}

/*
Gemm implementation:
Gemm:
  Inputs:
    0) A - Input Tensor
    1) B - Input Tensor
    2) C - Input Tensor (optional if Opset is 11 or later)
  Outputs:
    0) Y - Output Tensor

               +-----------+
    (A)        |           |
    ---------->+           |     AB               +------+
    (B)        | MatMul    +--------------------->+      | alphaAB
    ---------->+           |     (alpha)          | Mul  +---+
               |           |     *--------------->+      |   |     +------+
               +-----------+                      +------+   +---->+      |   (Y) alphaAB + betaC
                                                                   | Add  +---------------------->
    (C)                                           +------+   +---->+      |
    --------------------------------------------->+      |   |     +------+
                                 (beta)           | Mul  +---+
                                 *--------------->+      | betaC
                                                  +------+

Attributes (alpha, beta, transA, transB)

To compose Gemm: (algorithm)
(1) perform `MatMul` on input tensors A and B result (AB)
(2) if `Mul` the result of (1) by alpha attribute (alphaAB)
(3) if C is optional return result from (2) and end
(4) if C is avalible `Mul` input C tensor by beta attribute (betaC)
(5) `Add` result from (2) to result from (4) (alphaAB + betaC)
(6) Return output from (5) and end

ZenDNN algorithm:
(1) perform `MatMul` of tensor A and tensor B with `Output scales` set to alpha (0)
(2) if C is optional return output from (1) and end
(3) if C is avalible perform binary `Add` of output from (0) and input C with input C's `scale` attribute set to beta
(4) return output from (4) and end

*/


void ZendnnGemm::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                 ZendnnNode &node) {
    auto eng = sp.GetEngine();

    auto a_dims = sp.GetMemory(node.Input(IN_A)).get_desc().dims();
    auto b_dims = sp.GetMemory(node.Input(IN_B)).get_desc().dims();

    bool input_c_exists = node.Input(IN_C).Exists();

    if (a_dims.size() != b_dims.size()) {
        while (a_dims.size() < b_dims.size()) {
            a_dims.insert(a_dims.begin(), 1);
        }
        while (a_dims.size() > b_dims.size()) {
            b_dims.insert(b_dims.begin(), 1);
        }
    }


    zendnn::memory::desc a_md;
    zendnn::memory::desc b_md;

    bool transA = GetTransA(node);
    bool transB = GetTransB(node);

    zendnn::memory::dim M = (transA) ? a_dims[1] : a_dims[0];
    zendnn::memory::dim K = (transA) ? a_dims[0] : a_dims[1];
    zendnn::memory::dim N = (transB) ? b_dims[0] : b_dims[1];

    zendnn::memory::dims a_strides = (transA) ? zendnn::memory::dims{zendnn::memory::dim(1), M} :
                                     zendnn::memory::dims{K, zendnn::memory::dim(1)};
    zendnn::memory::dims b_strides = (transB) ? zendnn::memory::dims{zendnn::memory::dim(1), K} :
                                     zendnn::memory::dims{N, zendnn::memory::dim(1)};

    a_md = zendnn::memory::desc({M, K}, node.Input(IN_A).Type(), a_strides);
    b_md = zendnn::memory::desc({K, N}, node.Input(IN_B).Type(), b_strides);

    zendnn::memory::dims output_shape{M, N};

    zendnn::primitive_attr matmul_attr;
    // scale the output from MatMul to alpha
    float alpha = GetAlpha(node);
    std::vector<float> alphaScale({alpha});
    matmul_attr.set_output_scales(0, alphaScale);

    auto matmul_dst_md = zendnn::memory::desc(output_shape,
                         node.Output(OUT_Y).Type(), {N, 1});

    auto matmul_d = zendnn::matmul::desc(a_md, b_md, matmul_dst_md);
    zendnn::matmul::primitive_desc matmul_pd;
    matmul_pd = zendnn::matmul::primitive_desc(matmul_d, matmul_attr, eng);

    auto matmul_a_mem = sp.GetMemoryAndReshape(node.Input(IN_A),
                        matmul_pd.src_desc(), eng, transA);
    auto matmul_b_mem = sp.GetMemoryAndReshape(node.Input(IN_B),
                        matmul_pd.weights_desc(), eng, transB);
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    PrimitiveMemInfo mem_info;
    mem_info.ref_count  = out_links;
    mem_info.mem_desc   = matmul_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;
    zendnn::memory gemm_dst_mem;

    if (mem_info.is_dynamic) {
        gemm_dst_mem = zendnn::memory(matmul_pd.dst_desc(), eng, NULL);
    }
    else {
        gemm_dst_mem = zendnn::memory(matmul_pd.dst_desc(), eng);
    }

    auto matmul_op = zendnn::matmul(matmul_pd);

    std::unordered_map<int, zendnn::memory> args;
    args.insert({ZENDNN_ARG_SRC, matmul_a_mem});
    args.insert({ZENDNN_ARG_WEIGHTS, matmul_b_mem});
    args.insert({ZENDNN_ARG_DST, gemm_dst_mem});

    sp.AddPrimitive(matmul_op, args, mem_info);

    if (input_c_exists) {
        auto c_original_md = sp.GetMemory(node.Input(IN_C)).get_desc();
        auto c_dims = c_original_md.dims();
        if (c_dims.size() != a_dims.size()) {
            while (c_dims.size() < a_dims.size()) {
                c_dims.insert(c_dims.begin(), 1);
            }
        }

        auto c_md = c_original_md.reshape(c_dims);

        auto y_md = zendnn::memory::desc(output_shape, node.Output(OUT_Y).Type(),
                                         zendnn::memory::format_tag::any);

        auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_add,
                                             matmul_pd.dst_desc(), c_md, y_md);

        // Scale input C by beta before adding it to the MatMul output.
        zendnn::primitive_attr binary_attr;
        float beta = GetBeta(node);
        binary_attr.set_scales(ZENDNN_ARG_SRC_1, 0, {beta});

        auto binary_pd = zendnn::binary::primitive_desc(binary_d, binary_attr,eng);

        auto binary_c_mem = sp.GetMemoryAndReshape(node.Input(IN_C),
                            binary_pd.src1_desc(), eng);

        auto binary_op = zendnn::binary(binary_pd);

        sp.IncMemoryRefCount(gemm_dst_mem);

        sp.AddPrimitive(binary_op, {{ZENDNN_ARG_SRC_0, gemm_dst_mem},
            {ZENDNN_ARG_SRC_1, binary_c_mem},
            {ZENDNN_ARG_DST, gemm_dst_mem}
        });
    }
    sp.SetMemory(node.Output(OUT_Y), gemm_dst_mem);
}

float ZendnnGemm::GetAlpha(ZendnnNode &node) {
    auto attr = node.Attributes().find("alpha");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return 1.0;
}
float ZendnnGemm::GetBeta(ZendnnNode &node) {
    auto attr = node.Attributes().find("beta");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return 1.0;
}

bool ZendnnGemm::GetTransA(ZendnnNode &node) {
    auto attr = node.Attributes().find("transA");
    if (attr != node.Attributes().end()) {
        return (attr->second().i() != 0);
    }
    return false;
}

bool ZendnnGemm::GetTransB(ZendnnNode &node) {
    auto attr = node.Attributes().find("transB");
    if (attr != node.Attributes().end()) {
        return (attr->second().i() != 0);
    }
    return false;
}
}  // namespace ort_zendnn
}  // namespace onnxruntime
