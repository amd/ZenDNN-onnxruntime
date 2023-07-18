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

#include "zendnn_lrn.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnLrn::ZendnnLrn() {}

// assume all dims are available
void ZendnnLrn::CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node) {
    using namespace zendnn;

    // get the engine, currently only support either single gpu or single cpu device
    auto zendnn_engine = sp.GetEngine();

    auto alpha = ReadAlpha(node);
    auto beta = ReadBeta(node);
    auto bias = ReadBias(node);
    auto size = ReadSize(node);

    auto lrn_src_mem = sp.GetMemory(node.Input(IN_X));
    auto lrn_src_md = lrn_src_mem.get_desc();

    auto lrn_desc = zendnn::lrn_forward::desc(zendnn::prop_kind::forward_scoring,
                    zendnn::algorithm::lrn_across_channels, lrn_src_md, size, alpha, beta, bias);
    auto lrn_pd = zendnn::lrn_forward::primitive_desc(lrn_desc, zendnn_engine);

    // If using GPU this will move the memory from the CPU to the GPU.
    lrn_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), lrn_pd.src_desc(),
                                         zendnn_engine);
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    PrimitiveMemInfo mem_info;
    mem_info.ref_count = out_links;
    mem_info.mem_desc  = lrn_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;

    zendnn::memory lrn_dst_mem;
    if (mem_info.is_dynamic) {
        lrn_dst_mem = zendnn::memory(lrn_pd.dst_desc(), zendnn_engine, NULL);
    }
    else {
        lrn_dst_mem = zendnn::memory(lrn_pd.dst_desc(), zendnn_engine);
    }

    auto lrn_op = zendnn::lrn_forward(lrn_pd);
    sp.AddPrimitive(lrn_op, {{ZENDNN_ARG_SRC, lrn_src_mem},
        {ZENDNN_ARG_DST, lrn_dst_mem}
    }, mem_info);

    sp.SetMemory(node.Output(OUT_Y), lrn_dst_mem);
}

int64_t ZendnnLrn::ReadSize(ZendnnNode &node) {
    auto attr = node.Attributes().find("size");
    int64_t size = 0;
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
        size = attr->second().i();
    }
    ORT_ENFORCE(size > 0);
    ORT_ENFORCE(size % 2 == 1);
    return size;
}

float ZendnnLrn::ReadAlpha(ZendnnNode &node) {
    auto attr = node.Attributes().find("alpha");
    float alpha = 0;
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
        alpha = attr->second().f();
    }
    return alpha;
}

float ZendnnLrn::ReadBeta(ZendnnNode &node) {
    auto attr = node.Attributes().find("beta");
    float beta = 0;
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
        beta = attr->second().f();
    }
    return beta;
}

float ZendnnLrn::ReadBias(ZendnnNode &node) {
    auto attr = node.Attributes().find("bias");
    float bias = 1.0f;
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
        bias = attr->second().f();
    }
    return bias;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
