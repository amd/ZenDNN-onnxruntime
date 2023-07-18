/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "zendnn_softmaxgrad.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnSoftmaxGrad::ZendnnSoftmaxGrad() {}

void ZendnnSoftmaxGrad::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                        ZendnnNode &node) {
    auto eng = sp.GetEngine();

    //get what's available as input
    auto src_mem = sp.GetMemory(node.Input(IN_X));
    auto diff_dst_mem = sp.GetMemory(node.Input(IN_dY));

    //reorder if needed (gpu)
    auto softmax_bwd_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X),
                               src_mem.get_desc(), eng);
    auto softmax_bwd_diff_dst_mem = sp.GetMemoryAndReshape(node.Input(IN_dY),
                                    diff_dst_mem.get_desc(), eng);

    auto axis = ReadAxis(node);

    if (axis < 0) {
        axis = src_mem.get_desc().dims().size() + axis;
    }

    //create hints on the fly
    auto hints_d = zendnn::softmax_forward::desc(
                       zendnn::prop_kind::forward_training, softmax_bwd_src_mem.get_desc(),
                       (int) axis);
    auto hints_pd = zendnn::softmax_forward::primitive_desc(hints_d, eng);

    auto softmax_bwd_d = zendnn::softmax_backward::desc(
                             softmax_bwd_diff_dst_mem.get_desc(), softmax_bwd_src_mem.get_desc(),
                             (int) axis);

    auto softmax_bwd_pd = zendnn::softmax_backward::primitive_desc(softmax_bwd_d,
                          eng, hints_pd);

    auto softmax_bwd_diff_src_mem = zendnn::memory(softmax_bwd_pd.diff_src_desc(),
                                    eng);

    auto softmax_bwd = zendnn::softmax_backward(softmax_bwd_pd);

    sp.AddPrimitive(softmax_bwd, {{ZENDNN_ARG_DST, softmax_bwd_src_mem},
        {ZENDNN_ARG_DIFF_DST, softmax_bwd_diff_dst_mem},
        {ZENDNN_ARG_DIFF_SRC, softmax_bwd_diff_src_mem}
    });

    sp.SetMemory(node.Output(OUT_dX), softmax_bwd_diff_src_mem);
}

int64_t ZendnnSoftmaxGrad::ReadAxis(ZendnnNode &node) {
    auto attr = node.Attributes().find("axis");
    int64_t axis =
        -1;  //Default value according to ONNX spec 13 but works with lower opsets too
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
        axis = attr->second().i();
    }
    return axis;
}
}  // namespace ort_zendnn
}  // namespace onnxruntime
