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

#include "zendnn_softmax.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {


ZendnnSoftmax::ZendnnSoftmax() {}

void ZendnnSoftmax::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                    ZendnnNode &node) {

    using namespace zendnn;

    // get the engine, currently only support either single gpu or single cpu device
    auto zendnn_engine = sp.GetEngine();

    auto axis = ReadAxis(node);

    auto softmax_src_mem = sp.GetMemory(node.Input(IN_X));
    auto softmax_src_md = softmax_src_mem.get_desc();

    if (axis < 0) {
        axis = softmax_src_md.dims().size() + axis;
    }

    auto softmax_desc = zendnn::softmax_forward::desc(
                            zendnn::prop_kind::forward_training, softmax_src_md, (int) axis);
    auto softmax_pd = zendnn::softmax_forward::primitive_desc(softmax_desc,
                      zendnn_engine);

    // If using GPU this will move the memory from the CPU to the GPU.
    softmax_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X),
                      softmax_pd.src_desc(), zendnn_engine);
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    zendnn::memory softmax_dst_mem;
    PrimitiveMemInfo mem_info;
    mem_info.ref_count = out_links;
    mem_info.mem_desc  = softmax_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;
    if (mem_info.is_dynamic) {
        softmax_dst_mem = zendnn::memory(softmax_pd.dst_desc(), zendnn_engine, NULL);
    }
    else {
        softmax_dst_mem = zendnn::memory(softmax_pd.dst_desc(), zendnn_engine);
    }

    auto softmax_op = zendnn::softmax_forward(softmax_pd);
    sp.AddPrimitive(softmax_op, {{ZENDNN_ARG_SRC, softmax_src_mem},
        {ZENDNN_ARG_DST, softmax_dst_mem}
    }, mem_info);
    if (sp.IsScalar(node.Input(IN_X))) {
        sp.SetMemory(node.Output(OUT_Y), softmax_dst_mem, false, true);
    }
    else {
        sp.SetMemory(node.Output(OUT_Y), softmax_dst_mem);
    }
}

int64_t ZendnnSoftmax::ReadAxis(ZendnnNode &node) {
    auto attr = node.Attributes().find("axis");
    int64_t axis =
        -1; //Default value according to ONNX spec 13 but works with lower opset too
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
        axis = attr->second().i();
    }
    return axis;
}


}  // namespace ort_zendnn
}  // namespace onnxruntime
