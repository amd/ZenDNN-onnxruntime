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

#include "zendnn_quantizelinear.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

/*
 y = saturate ((x / y_scale) + y_zero_point)
 For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
 For (x / y_scale), it's rounding to nearest ties to even.
 'y_zero_point' and 'y' must have same type.
*/
void ZendnnQuantizeLinear::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
        ZendnnNode &node) {

    using tag = zendnn::memory::format_tag;
    using dt =  zendnn::memory::data_type;

    auto zendnn_engine = sp.GetEngine();

    // Get the x and scale mem
    auto x_fp32_mem = sp.GetMemory(node.Input(IN_X));
    auto x_fp32_md = x_fp32_mem.get_desc();
    auto x_dims_size = x_fp32_md.dims().size();

    auto op_tag = tag::nchw;
    op_tag = sp.GetZendnnFormat(x_dims_size);

    auto y_scale_mem = sp.GetMemory(node.Input(IN_Y_SCALE));
    auto y_scale_md = y_scale_mem.get_desc();
    Padd(&y_scale_md, x_fp32_mem.get_desc().dims().size());
    auto y_scale_fp32_mem = zendnn::memory({y_scale_md.dims(), dt::f32, op_tag},
                                           zendnn_engine);
    y_scale_fp32_mem.set_data_handle(y_scale_mem.get_data_handle());

    auto opType = node.Output(OUT_Y).Type();

    auto dst_u8_mem = zendnn::memory({x_fp32_md.dims(), opType, op_tag},
                                     zendnn_engine);
    auto binary_desc = zendnn::binary::desc(zendnn::algorithm::binary_div,
                                            x_fp32_mem.get_desc(), y_scale_fp32_mem.get_desc(), dst_u8_mem.get_desc());

    auto y_zp_mem = sp.GetMemory(node.Input(IN_Y_ZERO_POINT));
    bool isZeroPointUseful = isZeroPointNonZero(&y_zp_mem);

    zendnn::post_ops binary_ops;
    zendnn::primitive_attr binary_attr;

    if (isZeroPointUseful) {
        auto y_zp_md = y_zp_mem.get_desc();
        Padd(&y_zp_md, x_fp32_mem.get_desc().dims().size());
        auto y_zp_u8_mem = zendnn::memory({y_zp_md.dims(), opType, op_tag},
                                          zendnn_engine);
        y_zp_u8_mem.set_data_handle(y_zp_mem.get_data_handle());

        binary_ops.append_binary(zendnn::algorithm::binary_add, y_zp_u8_mem.get_desc());
        binary_attr.set_post_ops(binary_ops);
        auto binary_pd = zendnn::binary::primitive_desc(binary_desc, binary_attr,
                         zendnn_engine);

        auto dst_mem = zendnn::memory(binary_pd.dst_desc(), zendnn_engine);

        sp.AddPrimitive(zendnn::binary(binary_pd), {{ZENDNN_ARG_SRC_0, x_fp32_mem},
            {ZENDNN_ARG_SRC_1, y_scale_fp32_mem},
            {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, y_zp_u8_mem},
            {ZENDNN_ARG_DST, dst_mem}
        });

        sp.SetMemory(node.Output(OUT_Y), dst_mem);

    }
    else {
        auto binary_pd = zendnn::binary::primitive_desc(binary_desc, zendnn_engine);

        auto dst_mem = zendnn::memory(binary_pd.dst_desc(), zendnn_engine);

        sp.AddPrimitive(zendnn::binary(binary_pd), {{ZENDNN_ARG_SRC_0, x_fp32_mem},
            {ZENDNN_ARG_SRC_1, y_scale_fp32_mem},
            {ZENDNN_ARG_DST, dst_mem}
        });

        sp.SetMemory(node.Output(OUT_Y), dst_mem);

    }

}

void ZendnnQuantizeLinear::Padd(zendnn::memory::desc *target_md, size_t pad) {
    // Pads an input to broadcast the op correctly
    auto target_dims = target_md->dims();   // Add back padd
    while (target_dims.size() < pad) {
        target_dims.insert(target_dims.end(), 1);
    }
    *target_md = target_md->reshape(target_dims);
}

bool ZendnnQuantizeLinear::isZeroPointNonZero(zendnn::memory *zp_mem) {
    // Because zp will always be int8, uint8 or int32, this cast is always valid
    auto zp_data = static_cast<uint8_t *>(zp_mem->get_data_handle());
    //  Adjust the iteration num
    auto topline = zp_mem->get_desc().dims().size();
    if (zp_mem->get_desc().data_type() == zendnn::memory::data_type::s32) {
        topline *= 4;
    }
    // ZP is either a scalar or a 1-D vector so iterate over all the dimensions
    // and search for a zp != 0
    for (size_t i = 0; i < topline; ++i) {
        if (zp_data[i] != 0) {
            return true;
        }
    }
    // If ZP is full of zeros then it is not needed
    return false;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime