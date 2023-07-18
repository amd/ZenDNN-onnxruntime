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

#include "zendnn_sum.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnSum::ZendnnSum() {}

void ZendnnSum::CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    std::vector<zendnn::memory> src_mems;
    for (size_t i = IN_DATA_0; i < node.InputCount(); ++i) {
        src_mems.push_back(sp.GetMemoryInOrtFormat(node.Input(static_cast<int>
                           (IN_DATA_0 + i)), zendnn_engine));
    }

    std::vector<float> scales;
    std::vector<zendnn::memory::desc> srcs_pd;
    for (size_t i = 0; i < src_mems.size(); ++i) {
        srcs_pd.push_back(src_mems[i].get_desc());
        scales.push_back(1.0f);
    }

    auto dst_dims = srcs_pd[0].dims();
    auto dst_md =  zendnn::memory::desc({dst_dims}, node.Input(IN_DATA_0).Type(),
                                        zendnn::memory::format_tag::any);

    auto sum_pd = zendnn::sum::primitive_desc(dst_md, scales, srcs_pd,
                  zendnn_engine);

    for (size_t i = 0; i < src_mems.size(); ++i) {
        src_mems[i] = sp.GetMemoryAndReshape(node.Input(static_cast<int>
                                             (IN_DATA_0 + i)), sum_pd.src_desc(), zendnn_engine);
    }
    auto sum_dst_mem = zendnn::memory(sum_pd.dst_desc(), zendnn_engine);

    auto sum_op = zendnn::sum(sum_pd);

    std::unordered_map<int, zendnn::memory> sum_args;
    sum_args.insert({ZENDNN_ARG_DST, sum_dst_mem});
    for (int i = 0; i < static_cast<int>(src_mems.size()); ++i) {
        sum_args.insert({ZENDNN_ARG_MULTIPLE_SRC + i, src_mems[i]});
    }

    sp.AddPrimitive(sum_op, sum_args);

    sp.SetMemory(node.Output(OUT_SUM), sum_dst_mem);
}

}  // namespace ort_zendnn
}  // namespace onnxruntime