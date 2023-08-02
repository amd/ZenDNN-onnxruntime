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

#include "zendnn_binary.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "zendnn_util.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnBinary::ZendnnBinary() {}

void ZendnnBinary::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                   ZendnnNode &node) {
    auto eng = sp.GetEngine();

    zendnn::algorithm algo = zendnn_util::OrtOperatorToZendnnAlgorithm(
                                 node.OpType());
    using dt = zendnn::memory::data_type;
    bool zendnn_enable_bf16 = false;
    const std::string enable_bf16_env = onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty())
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);

    // GetMemory in OrtFormat. Broadcasting and mix format binary ops can result in computation failure
    auto binary_src0_mem = sp.GetMemoryInOrtFormat(node.Input(IN_A), eng);
    auto binary_src1_mem = sp.GetMemoryInOrtFormat(node.Input(IN_B), eng);

    if(zendnn_enable_bf16)
    {
        zendnn::stream strm{eng};
        //Source 1
        if(binary_src0_mem.get_desc().data_type() != dt::bf16)
        {
            auto src_0_dims = binary_src0_mem.get_desc().dims();
            auto binary_src0_bf16_md = zendnn::memory::desc(src_0_dims,dt::bf16,sp.GetZendnnFormat(src_0_dims.size()));
            auto binary_src0_bf16_mem = sp.GetMemoryAndReshape(node.Input(IN_A),binary_src0_bf16_md, eng );
            binary_src0_mem = binary_src0_bf16_mem;
        }

        //source 2
        if(binary_src1_mem.get_desc().data_type() != dt::bf16)
        {
            auto src_1_dims = binary_src1_mem.get_desc().dims();
            auto binary_src1_bf16_md = zendnn::memory::desc(src_1_dims,dt::bf16,sp.GetZendnnFormat(src_1_dims.size()));
            auto binary_src1_bf16_mem = sp.GetMemoryAndReshape(node.Input(IN_B),binary_src1_bf16_md, eng );
            binary_src1_mem = binary_src1_bf16_mem;
        }
    }

    auto src_0_ori_md = binary_src0_mem.get_desc();
    auto src_1_ori_md = binary_src1_mem.get_desc();

    auto src_0_dims = src_0_ori_md.dims();
    auto src_1_dims = src_1_ori_md.dims();
    if (src_0_dims.size() != src_1_dims.size()) {
        while (src_0_dims.size() < src_1_dims.size()) {
            src_0_dims.insert(src_0_dims.begin(), 1);
        }
        while (src_0_dims.size() > src_1_dims.size()) {
            src_1_dims.insert(src_1_dims.begin(), 1);
        }
    }

    auto src_0_md = src_0_ori_md.reshape(src_0_dims);
    auto src_1_md = src_1_ori_md.reshape(src_1_dims);

    auto output_shape = src_0_dims;
    for (size_t i = 0; i < output_shape.size(); i++) {
        if (output_shape[i] == 1) {
            output_shape[i] = src_1_dims[i];
        }
    }

    auto dst_md = zendnn::memory::desc(output_shape, node.Output(OUT_Y).Type(),
                                       zendnn::memory::format_tag::any);

    auto binary_d = zendnn::binary::desc(algo, src_0_md, src_1_md, dst_md);
    auto binary_pd = zendnn::binary::primitive_desc(binary_d, eng);

    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    PrimitiveMemInfo mem_info;
    mem_info.ref_count  = out_links;
    mem_info.mem_desc   = binary_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;

    zendnn::memory binary_dst_mem;

    if (mem_info.is_dynamic) {
        binary_dst_mem = zendnn::memory(binary_pd.dst_desc(), eng, NULL);
    }
    else {
        binary_dst_mem = zendnn::memory(binary_pd.dst_desc(), eng);
    }
    auto binary_prim = zendnn::binary(binary_pd);

    sp.AddPrimitive(binary_prim, {{ZENDNN_ARG_SRC_0, binary_src0_mem},
        {ZENDNN_ARG_SRC_1, binary_src1_mem},
        {ZENDNN_ARG_DST, binary_dst_mem}
    }, mem_info);

    if (sp.IsScalar(node.Input(IN_A)) && sp.IsScalar(node.Input(IN_B))) {
        sp.SetMemory(node.Output(OUT_Y), binary_dst_mem, false, true);
    }
    else {
        sp.SetMemory(node.Output(OUT_Y), binary_dst_mem);
    }
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
