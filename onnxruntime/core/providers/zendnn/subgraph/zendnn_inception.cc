/*******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "zendnn_inception.h"
#include "zendnn_subgraph.h"
#include "zendnn_conv.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnInception::ZendnnInception() {}

void ZendnnInception::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                      ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();
    bool in_place = true;
    zendnn::memory::dims dst_mem_dims;
    auto axis_dim = 0 /*, tmp_offset = 0*/;
    std::vector<zendnn::memory::dims> ldims, loffset;
    auto concat_axis = GetAxis(node);
    int out_links = (int)node.Output(OUT_ZEN_INCEPTION).GetConsumersCount();
    for (size_t i = IN_DATA_0; i < node.InputCount(); ++i) {
        auto convNode = node.Input((int)(IN_DATA_0 + i)).GetProducer().GetNode();
        ZendnnConv zen_conv;
        zen_conv.GetDestMemoryDims(sp, *convNode, dst_mem_dims);
        for (size_t j = 0; j < dst_mem_dims.size(); j++) {
            if (j == (size_t)concat_axis && dst_mem_dims[j] % 8) {
                in_place = false;
                break;
            }
        }
        if (in_place == false) {
            break;
        }
        ldims.push_back(dst_mem_dims);
        loffset.push_back(zendnn::memory::dims(dst_mem_dims.size(), 0));
        loffset[i][concat_axis] = axis_dim;
        axis_dim += (int)dst_mem_dims[concat_axis];
    }
    if (in_place == true) {
        dst_mem_dims[concat_axis] = axis_dim;
        auto dst_md = zendnn::memory::desc({dst_mem_dims,
                                            zendnn::memory::data_type::f32,
                                            zendnn::memory::format_tag::aBcd8b});
        zendnn::memory dst_mem;
        if (out_links > 0 && sp.UseCPUAllocator()) {
            dst_mem = zendnn::memory(dst_md, zendnn_engine, NULL);
        }
        else {
            dst_mem = zendnn::memory(dst_md, zendnn_engine);
        }

        for (size_t i = IN_DATA_0; i < node.InputCount(); ++i) {
            auto convNode = node.Input((int)(IN_DATA_0 + i)).GetProducer().GetNode();
            ZendnnConv zen_conv;
            zendnn::memory::desc ldst_md = dst_md.submemory_desc(ldims[i], loffset[i]);
            zen_conv.SetDestinationMemoryInfo(ldst_md, dst_mem, (int)i, out_links,  false);
            zen_conv.CreatePrimitive(sp, *convNode);
        }
        sp.SetMemory(node.Output(OUT_ZEN_INCEPTION), dst_mem);
    }
    else {
        std::vector<zendnn::memory> src_mems;
        std::vector<zendnn::memory::desc> srcs_md;
        for (size_t i = IN_DATA_0; i < node.InputCount(); ++i) {
            auto convNode = node.Input((int)(IN_DATA_0 + i)).GetProducer().GetNode();
            ZendnnConv zen_conv;
            zen_conv.GetDestMemoryDims(sp, *convNode, dst_mem_dims);
            auto dst_md = zendnn::memory::desc({dst_mem_dims,
                                                zendnn::memory::data_type::f32,
                                                zendnn::memory::format_tag::aBcd8b});
            zendnn::memory dst_mem = zendnn::memory(dst_md, zendnn_engine);
            src_mems.push_back(dst_mem);
            srcs_md.push_back(dst_md);
            zen_conv.SetDestinationMemoryInfo(dst_md, dst_mem,(int)i, out_links, true);
            zen_conv.CreatePrimitive(sp, *convNode);
        }
        if (concat_axis < 0) {
            concat_axis = srcs_md.at(0).dims().size() + concat_axis;
        }

        auto concat_pd = zendnn::concat::primitive_desc((int)concat_axis, srcs_md,
                         zendnn_engine);
        auto concat_dst_mem = zendnn::memory(concat_pd.dst_desc(), zendnn_engine);
        auto concat_op = zendnn::concat(concat_pd);
        std::unordered_map<int, zendnn::memory> concat_args;
        concat_args.insert({ZENDNN_ARG_DST, concat_dst_mem});
        for (int i = 0; i < static_cast<int>(src_mems.size()); ++i) {
            concat_args.insert({ZENDNN_ARG_MULTIPLE_SRC + i, src_mems[i]});
        }
        sp.AddPrimitive(concat_op, concat_args);
        sp.SetMemory(node.Output(OUT_ZEN_INCEPTION), concat_dst_mem);
    }
}

int64_t ZendnnInception::GetAxis(ZendnnNode &node) {
    auto attr = node.Attributes().find("axis");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return 1;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
