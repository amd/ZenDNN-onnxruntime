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

#include <float.h>
#include "zendnn_elementwise.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "zendnn_util.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnElementwise::ZendnnElementwise() {}

void ZendnnElementwise::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                        ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    auto elementwise_src_mem = sp.GetMemory(node.Input(IN_X));
    auto src_md = elementwise_src_mem.get_desc();
    zendnn::algorithm algo = zendnn_util::OrtOperatorToZendnnAlgorithm(
                                 node.OpType());
    bool requires_alpha = false;
    bool requires_beta = false;
    float alpha = 0.0;
    float beta = 0.0;
    switch (algo) {
    case zendnn::algorithm::eltwise_elu: {
        requires_alpha = true;
        alpha = GetAlpha(node, /*default_alpha*/ 1.0f);
        break;
    }
    case zendnn::algorithm::eltwise_relu: {
        // Need to check operator since both Relu and LeakyRelu are covered by algorithm::eltwise_relu
        if (node.OpType() == "LeakyRelu") {
            requires_alpha = true;
            alpha = GetAlpha(node, /*default_alpha*/ 0.01f);
        }
        else {
            alpha = 0.0;
        }
        break;
    }
    default:
        alpha = 0.0;
    }

    if (node.OpType() == "Clip") {
        requires_alpha = true;
        requires_beta = true;
        alpha = -(FLT_MAX);
        beta = FLT_MAX;
        if (node.InputCount() > 1) {
            if (node.Input(IN_ALPHA).Exists()) {
                auto mem_alpha = sp.GetMemory(node.Input(IN_ALPHA));
                if (mem_alpha.get_desc().get_size() != 0) {
                    alpha = *((float *)mem_alpha.get_data_handle());
                }
            }
            if (node.Input(IN_BETA).Exists()) {
                auto mem_beta  = sp.GetMemory(node.Input(IN_BETA));
                if (mem_beta.get_desc().get_size() != 0) {
                    beta  = *((float *)mem_beta.get_data_handle());
                }
            }
        }
        else {
            alpha = GetMin(node, /*default_alpha*/-(FLT_MAX));
            beta  = GetMax(node, /*default_alpha*/FLT_MAX);
        }
    }

    zendnn::eltwise_forward::primitive_desc elementwise_pd;
    if (requires_alpha && requires_beta) {
        auto elementwise_desc = zendnn::eltwise_forward::desc(
                                    zendnn::prop_kind::forward_inference, algo, src_md, alpha, beta);
        elementwise_pd = zendnn::eltwise_forward::primitive_desc(elementwise_desc,
                         zendnn_engine);
    }
    else if (requires_alpha) {
        auto elementwise_desc = zendnn::eltwise_forward::desc(
                                    zendnn::prop_kind::forward_inference, algo, src_md, alpha);
        elementwise_pd = zendnn::eltwise_forward::primitive_desc(elementwise_desc,
                         zendnn_engine);
    }
    else {
        auto elementwise_desc = zendnn::eltwise_forward::desc(
                                    zendnn::prop_kind::forward_inference, algo, src_md);
        elementwise_pd = zendnn::eltwise_forward::primitive_desc(elementwise_desc,
                         zendnn_engine);
    }

    // If using GPU this will move the memory from the CPU to the GPU.
    elementwise_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X),
                          elementwise_pd.src_desc(), zendnn_engine);
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    PrimitiveMemInfo mem_info;
    mem_info.ref_count  = out_links;
    mem_info.mem_desc   = elementwise_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;
    zendnn::memory elementwise_dst_mem;
    if (mem_info.is_dynamic) {
        elementwise_dst_mem  = zendnn::memory(elementwise_pd.dst_desc(),
                                              zendnn_engine, NULL);
    }
    else {
        elementwise_dst_mem  = zendnn::memory(elementwise_pd.dst_desc(),
                                              zendnn_engine);
    }

    auto elemenwise_primitive = zendnn::eltwise_forward(elementwise_pd);
    sp.AddPrimitive(elemenwise_primitive, {{ZENDNN_ARG_SRC, elementwise_src_mem},
        {ZENDNN_ARG_DST, elementwise_dst_mem}
    }, mem_info);
    if (sp.IsScalar(node.Input(IN_X))) {
        sp.SetMemory(node.Output(OUT_Y), elementwise_dst_mem, false, true);
    }
    else {
        sp.SetMemory(node.Output(OUT_Y), elementwise_dst_mem);
    }
}

float ZendnnElementwise::GetAlpha(ZendnnNode &node, float default_alpha) {
    auto attr = node.Attributes().find("alpha");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return default_alpha;
}

float ZendnnElementwise::GetMin(ZendnnNode &node, float default_min) {
    auto attr = node.Attributes().find("min");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return default_min;
}

float ZendnnElementwise::GetMax(ZendnnNode &node, float default_max) {
    auto attr = node.Attributes().find("max");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return default_max;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
