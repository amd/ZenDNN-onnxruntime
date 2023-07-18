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

#include "zendnn_relugrad.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnReluGrad::ZendnnReluGrad() {}

void ZendnnReluGrad::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                     ZendnnNode &node) {
    auto eng = sp.GetEngine();

    //get what's available as input
    auto src_mem = sp.GetMemory(node.Input(IN_X));
    auto diff_dst_mem = sp.GetMemory(node.Input(IN_dY));

    //reorder if needed (gpu)
    auto relu_bwd_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X),
                            src_mem.get_desc(), eng);
    auto relu_bwd_diff_dst_mem = sp.GetMemoryAndReshape(node.Input(IN_dY),
                                 diff_dst_mem.get_desc(), eng);

    //create hints on the fly
    auto hints_d = zendnn::eltwise_forward::desc(zendnn::prop_kind::forward,
                   zendnn::algorithm::eltwise_relu, relu_bwd_src_mem.get_desc(), 0.0, 0.0);
    auto hints_pd = zendnn::eltwise_forward::primitive_desc(hints_d, eng);

    auto relu_bwd_d = zendnn::eltwise_backward::desc(
                          zendnn::algorithm::eltwise_relu, relu_bwd_diff_dst_mem.get_desc(),
                          relu_bwd_src_mem.get_desc(), 0.0, 0.0);

    auto relu_bwd_pd = zendnn::eltwise_backward::primitive_desc(relu_bwd_d, eng,
                       hints_pd);

    auto relu_bwd_diff_src_mem = zendnn::memory(relu_bwd_pd.diff_src_desc(), eng);

    auto relu_bwd = zendnn::eltwise_backward(relu_bwd_pd);

    sp.AddPrimitive(relu_bwd, {{ZENDNN_ARG_SRC, relu_bwd_src_mem},
        {ZENDNN_ARG_DIFF_DST, relu_bwd_diff_dst_mem},
        {ZENDNN_ARG_DIFF_SRC, relu_bwd_diff_src_mem}
    });

    sp.SetMemory(node.Output(OUT_dX), relu_bwd_diff_src_mem);
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
