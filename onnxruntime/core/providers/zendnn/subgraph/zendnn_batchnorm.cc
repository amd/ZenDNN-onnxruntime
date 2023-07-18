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

#include "zendnn_batchnorm.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {


ZendnnBatchNorm::ZendnnBatchNorm() {}


void ZendnnBatchNorm::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                      ZendnnNode &node) {

    using namespace zendnn;

    // get the engine, currently only support either single gpu or single cpu device
    auto zendnn_engine = sp.GetEngine();

    auto epsilon = ReadEpsilon(node);

    auto batchnorm_src_mem = sp.GetMemory(node.Input(IN_X));
    auto src_md = batchnorm_src_mem.get_desc();

    auto batchnorm_scale_mem = sp.GetMemory(node.Input(IN_SCALE));
    auto scale_md = batchnorm_scale_mem.get_desc();
    auto scale_dims = scale_md.dims();

    auto batchnorm_bias_mem = sp.GetMemory(node.Input(IN_B));
    auto bias_md = batchnorm_bias_mem.get_desc();

    auto batchnorm_mean_mem = sp.GetMemory(node.Input(IN_MEAN));
    auto mean_md = batchnorm_mean_mem.get_desc();

    auto batchnorm_var_mem = sp.GetMemory(node.Input(IN_VAR));
    auto var_md = batchnorm_var_mem.get_desc();


    std::vector<memory::desc> src_mds;
    src_mds.push_back(scale_md);
    src_mds.push_back(bias_md);
    const int axis = 0;

    //To make the inputs compatible with ZenDNN, we need to concatenate scale and bias into a single tensor of length 2XC
    //Then, we create the batchnorm pd and feed in the inputs.
    auto concat_pd = zendnn::concat::primitive_desc(axis, src_mds, zendnn_engine);

    //If using GPU this will move the memory from the CPU to the GPU.
    batchnorm_scale_mem = sp.GetMemoryAndReshape(node.Input(IN_SCALE),
                          concat_pd.src_desc(), zendnn_engine);
    batchnorm_bias_mem = sp.GetMemoryAndReshape(node.Input(IN_B),
                         concat_pd.src_desc(), zendnn_engine);
    batchnorm_mean_mem = sp.GetMemoryAndReshape(node.Input(IN_MEAN), mean_md,
                         zendnn_engine);
    batchnorm_var_mem = sp.GetMemoryAndReshape(node.Input(IN_VAR), var_md,
                        zendnn_engine);
    auto batchnorm_scale_shift_mem = zendnn::memory(concat_pd.dst_desc(),
                                     zendnn_engine);

    zendnn::primitive_attr attr;
    if (node.OpType() == "BatchnormRelu") {
        const float ops_scale = 1.f;
        const float ops_alpha = 0.f;
        const float ops_beta = 0.f;
        zendnn::post_ops ops;

        ops.append_eltwise(ops_scale, zendnn::algorithm::eltwise_relu, ops_alpha,
                           ops_beta);
        attr.set_post_ops(ops);
    }

    auto batchnorm_desc = zendnn::batch_normalization_forward::desc(
                              zendnn::prop_kind::forward_inference, src_md, epsilon,
                              zendnn::normalization_flags::use_scale_shift |
                              zendnn::normalization_flags::use_global_stats);
    auto batchnorm_pd = zendnn::batch_normalization_forward::primitive_desc(
                            batchnorm_desc, attr, zendnn_engine);

    // If using GPU this will move the memory from the CPU to the GPU.
    batchnorm_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X),
                        batchnorm_pd.src_desc(), zendnn_engine);
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    PrimitiveMemInfo mem_info;
    mem_info.ref_count = out_links;
    mem_info.mem_desc  = batchnorm_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;

    zendnn::memory batchnorm_dst_mem;
    if (mem_info.is_dynamic) {
        batchnorm_dst_mem = zendnn::memory(batchnorm_pd.dst_desc(), zendnn_engine,
                                           NULL);
    }
    else {
        batchnorm_dst_mem = zendnn::memory(batchnorm_pd.dst_desc(), zendnn_engine);
    }

    auto concat_op = zendnn::concat(concat_pd);
    sp.AddPrimitive(concat_op, {{ZENDNN_ARG_MULTIPLE_SRC, batchnorm_scale_mem},
        {ZENDNN_ARG_MULTIPLE_SRC+1, batchnorm_bias_mem},
        {ZENDNN_ARG_DST, batchnorm_scale_shift_mem}
    });

    auto batchnorm_op = zendnn::batch_normalization_forward(batchnorm_pd);
    sp.AddPrimitive(batchnorm_op, {{ZENDNN_ARG_SRC, batchnorm_src_mem},
        {ZENDNN_ARG_MEAN, batchnorm_mean_mem},
        {ZENDNN_ARG_VARIANCE, batchnorm_var_mem},
        {ZENDNN_ARG_SCALE_SHIFT, batchnorm_scale_shift_mem},
        {ZENDNN_ARG_DST, batchnorm_dst_mem}
    }, mem_info);

    sp.SetMemory(node.Output(OUT_Y), batchnorm_dst_mem);
}

float ZendnnBatchNorm::ReadEpsilon(ZendnnNode &node) {
    auto attr = node.Attributes().find("epsilon");
    float epsilon = 1e-05f;  //Default value according to ONNX spec
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
        epsilon = attr->second().f();
    }
    return epsilon;
}


}  // namespace ort_zendnn
}  // namespace onnxruntime
