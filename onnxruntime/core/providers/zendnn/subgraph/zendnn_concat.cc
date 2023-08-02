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

#include "zendnn_concat.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnConcat::ZendnnConcat() {}

void ZendnnConcat::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                   ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    using dt =  zendnn::memory::data_type;
    bool zendnn_enable_bf16 = false;
    const std::string enable_bf16_env = onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty())
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);

    std::vector<zendnn::memory> src_mems;
    for (size_t i = IN_DATA_0; i < node.InputCount(); ++i) {
        src_mems.push_back(sp.GetMemoryInOrtFormat(node.Input(static_cast<int>
                           (IN_DATA_0 + i)),zendnn_engine));
    }

    std::vector<zendnn::memory::desc> srcs_md;
    for (size_t i = 0; i < src_mems.size(); ++i) {
        auto data_type=src_mems[i].get_desc().data_type();
        if(zendnn_enable_bf16) {
            if(data_type != dt::bf16) {
                auto dst_bf16_desc = zendnn::memory::desc(src_mems[i].get_desc().dims(),
                                        dt::bf16,sp.GetZendnnFormat(src_mems[i].get_desc().dims().size()));
                auto dst_bf16_mem = zendnn::memory(dst_bf16_desc, zendnn_engine);
                sp.AddPrimitive(zendnn::reorder(src_mems[i],dst_bf16_mem), {{ZENDNN_ARG_SRC, src_mems[i]},
                                                                          {ZENDNN_ARG_DST, dst_bf16_mem}});
                srcs_md.push_back(src_mems[i].get_desc());
            }
            else {
                srcs_md.push_back(src_mems[i].get_desc());
            }
        }
        else {
            srcs_md.push_back(src_mems[i].get_desc());
        }       
    }

    int64_t axis = GetAxis(node);
    if (axis < 0) {
        axis = srcs_md.at(0).dims().size() + axis;
    }

    auto concat_pd = zendnn::concat::primitive_desc((int)axis, srcs_md,
                     zendnn_engine);
    int out_links = (int)node.Output(OUT_CONCAT).GetConsumersCount();
    PrimitiveMemInfo mem_info;
    mem_info.variable_inputs = (int)node.InputCount();
    mem_info.ref_count = out_links;
    mem_info.mem_desc  = concat_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;

    zendnn::memory concat_dst_mem;

    if (mem_info.is_dynamic) {
        concat_dst_mem = zendnn::memory(concat_pd.dst_desc(), zendnn_engine, NULL);
    }
    else {
        concat_dst_mem = zendnn::memory(concat_pd.dst_desc(), zendnn_engine);
    }
    auto concat_op = zendnn::concat(concat_pd);

    std::unordered_map<int, zendnn::memory> concat_args;
    concat_args.insert({ZENDNN_ARG_DST, concat_dst_mem});
    for (int i = 0; i < static_cast<int>(src_mems.size()); ++i) {
        concat_args.insert({ZENDNN_ARG_MULTIPLE_SRC + i, src_mems[i]});
    }

    sp.AddPrimitive(concat_op, concat_args, mem_info);

    sp.SetMemory(node.Output(OUT_CONCAT), concat_dst_mem);
}

int64_t ZendnnConcat::GetAxis(ZendnnNode &node) {
    auto attr = node.Attributes().find("axis");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return 1;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
