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

#include "zendnn_gelu.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "zendnn_util.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnGelu::ZendnnGelu() {}

void ZendnnGelu::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                 ZendnnNode &node) {

    auto zendnn_engine = sp.GetEngine();

    bool is_biased = node.Input(IN_BIAS).Exists();
    zendnn::memory src_mem;
    if (is_biased) {
        src_mem = sp.GetMemoryInOrtFormat(node.Input(IN_X), zendnn_engine);
    }
    else {
        src_mem = sp.GetMemory(node.Input(IN_X));
    }
    auto gelu_src_mem = src_mem;
    zendnn::memory dst_mem;
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    if (is_biased) {
        auto bias_mem = sp.GetMemoryInOrtFormat(node.Input(IN_BIAS), zendnn_engine);
        auto src0_ori_md = src_mem.get_desc();
        auto src1_ori_md = bias_mem.get_desc();

        auto src0_dims = src0_ori_md.dims();
        auto src1_dims = src1_ori_md.dims();
        if (src0_dims.size() != src1_dims.size()) {
            while (src0_dims.size() < src1_dims.size()) {
                src0_dims.insert(src0_dims.begin(), 1);
            }
            while (src0_dims.size() > src1_dims.size()) {
                src1_dims.insert(src1_dims.begin(), 1);
            }
        }

        auto src0_md = src0_ori_md.reshape(src0_dims);
        auto src1_md = src1_ori_md.reshape(src1_dims);

        auto output_shape = src0_dims;
        for (size_t i = 0; i < output_shape.size(); i++) {
            if (output_shape[i] == 1) {
                output_shape[i] = src1_dims[i];
            }
        }

        zendnn::primitive_attr attr;
        zendnn::post_ops ops;
        zendnn::algorithm algo = zendnn_util::OrtOperatorToZendnnAlgorithm(
                                     node.OpType());
        ops.append_eltwise(1.0f, algo, 1.0f, 1.0f);
        attr.set_post_ops(ops);

        auto dst_md = zendnn::memory::desc(output_shape, node.Output(OUT_Y).Type(),
                                           zendnn::memory::format_tag::any);

        auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_add, src0_md,
                                             src1_md, dst_md);
        auto binary_pd = zendnn::binary::primitive_desc(binary_d, attr, zendnn_engine);

        PrimitiveMemInfo mem_info;
        mem_info.ref_count = out_links;
        mem_info.mem_desc  = binary_pd.dst_desc();
        mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;
        if (mem_info.is_dynamic) {
            dst_mem = zendnn::memory(binary_pd.dst_desc(), zendnn_engine, NULL);
        }
        else {
            dst_mem = zendnn::memory(binary_pd.dst_desc(), zendnn_engine);
        }

        auto binary_prim = zendnn::binary(binary_pd);

        sp.AddPrimitive(binary_prim, {{ZENDNN_ARG_SRC_0, src_mem},
            {ZENDNN_ARG_SRC_1, bias_mem},
            {ZENDNN_ARG_DST, dst_mem}
        }, mem_info);
    }
    else {
        zendnn::algorithm algo = zendnn_util::OrtOperatorToZendnnAlgorithm(
                                     node.OpType());
        auto gelu_desc = zendnn::eltwise_forward::desc(
                             zendnn::prop_kind::forward_inference, algo, gelu_src_mem.get_desc());
        auto gelu_pd = zendnn::eltwise_forward::primitive_desc(gelu_desc,
                       zendnn_engine);

        // If using GPU this will move the memory from the CPU to the GPU.
        gelu_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), gelu_pd.src_desc(),
                                              zendnn_engine);

        PrimitiveMemInfo mem_info;
        mem_info.ref_count = out_links;
        mem_info.mem_desc  = gelu_pd.dst_desc();
        mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;
        if (mem_info.is_dynamic) {
            dst_mem = zendnn::memory(gelu_pd.dst_desc(), zendnn_engine, NULL);
        }
        else {
            dst_mem = zendnn::memory(gelu_pd.dst_desc(), zendnn_engine);
        }

        auto gelu_op = zendnn::eltwise_forward(gelu_pd);
        sp.AddPrimitive(gelu_op, {{ZENDNN_ARG_SRC, gelu_src_mem},
            {ZENDNN_ARG_DST, dst_mem}
        }, mem_info);
    }

    if (sp.IsScalar(node.Input(IN_X))) {
        sp.SetMemory(node.Output(OUT_Y), dst_mem, false, true);
    }
    else {
        sp.SetMemory(node.Output(OUT_Y), dst_mem);
    }
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
