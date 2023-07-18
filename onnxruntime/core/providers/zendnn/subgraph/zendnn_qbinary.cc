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

#include "zendnn_qbinary.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "zendnn_util.h"
#include <cmath>

namespace onnxruntime {
namespace ort_zendnn {

ZendnnQBinary::ZendnnQBinary() {}

void ZendnnQBinary::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                    ZendnnNode &node) {
    using tag = zendnn::memory::format_tag;
    using dt =  zendnn::memory::data_type;

    auto zendnn_engine = sp.GetEngine();

    zendnn::algorithm algo = zendnn_util::OrtOperatorToZendnnAlgorithm(
                                 node.OpType());

    // GetMemory in OrtFormat. Broadcasting and mix format binary ops can result in computation failure
    auto src0_mem = sp.GetMemoryInOrtFormat(node.Input(IN_A), zendnn_engine);
    auto s0_scale = sp.GetMemory(node.Input(IN_A_SCALE));
    auto s0_zp = sp.GetMemory(node.Input(IN_A_ZERO_POINT));

    auto src1_mem = sp.GetMemoryInOrtFormat(node.Input(IN_B), zendnn_engine);
    auto s1_scale = sp.GetMemory(node.Input(IN_B_SCALE));
    auto s1_zp = sp.GetMemory(node.Input(IN_B_ZERO_POINT));

    auto d_scale = sp.GetMemory(node.Input(IN_C_SCALE));
    auto d_zp = sp.GetMemory(node.Input(IN_C_ZERO_POINT));

    auto src_0_ori_md = src0_mem.get_desc();
    auto src_1_ori_md = src1_mem.get_desc();

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

    //src0_reorder (dequantize_linear; u8 to f32)
    zendnn::primitive_attr src0_reorder;
    std::vector<float> s0_s(1);
    s0_s[0] = (*(float *)(s0_scale.get_data_handle()));
    src0_reorder.set_output_scales(0, s0_s);
    std::vector<int> s0_z(1);
    if (node.Input(IN_A_ZERO_POINT).Type() == dt::u8) {
        s0_z[0] = (int)(*(uint8_t *)(s0_zp.get_data_handle()));
    }
    else if (node.Input(IN_A_ZERO_POINT).Type() == dt::s8) {
        s0_z[0] = (int)(*(int8_t *)(s0_zp.get_data_handle()));
    }
    else {
        LOGS_DEFAULT(ERROR) <<"Invalid. Non integer zero point detected";
    }
    src0_reorder.set_zero_points(ZENDNN_ARG_SRC, 0, s0_z);

    auto op_tag = tag::any;
    op_tag = sp.GetZendnnFormat(output_shape.size());

    zendnn::memory src0_f32_mem = zendnn::memory({{src_0_md.dims()}, dt::f32, op_tag},
    zendnn_engine);
    zendnn::primitive src0_r_prim = zendnn::reorder(src0_mem, src0_f32_mem,
                                    src0_reorder);
    std::unordered_map<int, zendnn::memory> src0_reorder_args;
    src0_reorder_args.insert({ZENDNN_ARG_SRC, src0_mem});
    src0_reorder_args.insert({ZENDNN_ARG_DST, src0_f32_mem});
    sp.AddPrimitive(src0_r_prim, src0_reorder_args);

    //src1_reorder (dequantize_linear; u8 to f32)
    zendnn::primitive_attr src1_reorder;
    std::vector<float> s1_s(1);
    s1_s[0] = (*(float *)(s1_scale.get_data_handle()));
    src1_reorder.set_output_scales(0, s1_s);
    std::vector<int> s1_z(1);
    if (node.Input(IN_B_ZERO_POINT).Type() == dt::u8) {
        s1_z[0] = (int)(*(uint8_t *)(s1_zp.get_data_handle()));
    }
    else if (node.Input(IN_B_ZERO_POINT).Type() == dt::s8) {
        s1_z[0] = (int)(*(int8_t *)(s1_zp.get_data_handle()));
    }
    else {
        LOGS_DEFAULT(ERROR) <<"Invalid. Non integer zero point detected";
    }

    src1_reorder.set_zero_points(ZENDNN_ARG_SRC, 0, s1_z);

    zendnn::memory src1_f32_mem = zendnn::memory(zendnn::memory::desc(
                                      src_1_md.dims(), dt::f32, op_tag), zendnn_engine);
    zendnn::primitive src1_r_prim = zendnn::reorder(src1_mem, src1_f32_mem,
                                    src1_reorder);
    std::unordered_map<int, zendnn::memory> src1_reorder_args;
    src1_reorder_args.insert({ZENDNN_ARG_SRC, src1_mem});
    src1_reorder_args.insert({ZENDNN_ARG_DST, src1_f32_mem});
    sp.AddPrimitive(src1_r_prim, src1_reorder_args);

    //binary(add, sub, mul, div) (f32)
    auto dst_dims = src_0_md.dims(); //can also be src_1_md.dims();
    zendnn::memory dst_fp32_mem = zendnn::memory(zendnn::memory::desc(dst_dims,
                                  dt::f32, op_tag), zendnn_engine);
    auto bin_desc = zendnn::binary::desc(algo, src0_f32_mem.get_desc(),
                                         src1_f32_mem.get_desc(), dst_fp32_mem.get_desc());
    auto bin_pd = zendnn::binary::primitive_desc(bin_desc, zendnn_engine);
    std::unordered_map<int, zendnn::memory> bin_args;
    bin_args.insert({ZENDNN_ARG_SRC_0, src0_f32_mem});
    bin_args.insert({ZENDNN_ARG_SRC_1, src1_f32_mem});
    bin_args.insert({ZENDNN_ARG_DST, dst_fp32_mem});
    auto binary_prim = zendnn::binary(bin_pd);
    sp.AddPrimitive(binary_prim, bin_args);

    //dst (quantize_linear; f32 to u8)
    auto user_dst_zp1_md = d_zp.get_desc();
    Padd(&user_dst_zp1_md,
         dst_fp32_mem.get_desc().dims().size());  //Need to provide broadcasted data
    auto user_dst_zp1_mem = zendnn::memory({user_dst_zp1_md.dims(), node.Input(IN_C_ZERO_POINT).Type(), op_tag},
                                           zendnn_engine);
    user_dst_zp1_mem.set_data_handle(d_zp.get_data_handle());

    auto user_dst_scale1_md = d_scale.get_desc();
    Padd(&user_dst_scale1_md,
         dst_fp32_mem.get_desc().dims().size());   //Need to provide broadcasted data
    auto user_dst_scale1_mem = zendnn::memory({user_dst_scale1_md.dims(), dt::f32, op_tag},
                               zendnn_engine);
    user_dst_scale1_mem.set_data_handle(d_scale.get_data_handle());

    auto dst_mem = zendnn::memory({dst_fp32_mem.get_desc().dims(), node.Input(IN_C_ZERO_POINT).Type(), op_tag},
                                  zendnn_engine);
    auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_div,
                                         dst_fp32_mem.get_desc(), user_dst_scale1_mem.get_desc(), dst_mem.get_desc());
    zendnn::post_ops binary_ops;
    zendnn::primitive_attr binary_attr;
    binary_ops.append_binary(zendnn::algorithm::binary_add,
                             user_dst_zp1_mem.get_desc());
    binary_attr.set_post_ops(binary_ops);
    auto binary_pd = zendnn::binary::primitive_desc(binary_d, binary_attr,
                     zendnn_engine);
    std::unordered_map<int, zendnn::memory> binql_args;
    binql_args.insert({ZENDNN_ARG_SRC_0, dst_fp32_mem});
    binql_args.insert({ZENDNN_ARG_SRC_1, user_dst_scale1_mem});
    binql_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, user_dst_zp1_mem});
    binql_args.insert({ZENDNN_ARG_DST, dst_mem});
    auto binary_ql_prim = zendnn::binary(binary_pd);
    sp.AddPrimitive(binary_ql_prim, binql_args);

    if (sp.IsScalar(node.Input(IN_A)) && sp.IsScalar(node.Input(IN_B))) {
        sp.SetMemory(node.Output(OUT_Y), dst_mem, false, true);
    }
    else {
        sp.SetMemory(node.Output(OUT_Y), dst_mem);
    }
}

void ZendnnQBinary::Padd(zendnn::memory::desc *target_md, size_t pad) {
    // Pads an input to broadcast the op correctly
    auto target_dims = target_md->dims();   // Add back padd
    while (target_dims.size() < pad) {
        target_dims.insert(target_dims.end(), 1);
    }
    *target_md = target_md->reshape(target_dims);
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
