/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright(C) 2022 Intel Corporation
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

#include "zendnn_dequantizelinear.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

/*
 y = (x - x_zero_point) * x_scale.
 'x_scale' and 'x_zero_point' must have same shape, and can be either a scalar
 for per-tensor or per layer quantization, or a 1-D tensor for per-axis quantization.
 'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape.
 In the case of dequantizing int32, there's no zero point (zero point is supposed to be 0).
*/
void ZendnnDequantizeLinear::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
        ZendnnNode &node) {
    // Get engine
    auto zendnn_engine = sp.GetEngine();

    // Validate dims and datatypes
    ValidateDims(sp, node);
    ValidateType(sp, node);

    // Check if scale and zp are scalars
    bool isScalar = sp.IsScalar(node.Input(IN_X_SCALE));
    // Check if zp is needed
    bool isZeroPointUseful = false;
    if (node.Input(IN_X_ZERO_POINT).Exists()) {
        // If zp exists then it's needed
        isZeroPointUseful = true;
        // If it's constant then we can evaluate if zp == 0
        if (node.Input(IN_X_ZERO_POINT).IsConstant()) {
            // if zp == 0 then isZeroPointUseful = false; else isZeroPointUseful = true
            auto mem = sp.GetMemory(node.Input(IN_X_ZERO_POINT));
            isZeroPointUseful = isZeroPointNonZero(&mem);
        }
    }

    // Get the x and scale mem
    auto x_mem = sp.GetMemory(node.Input(IN_X));
    auto x_scale_mem = sp.GetMemory(node.Input(IN_X_SCALE));
    // Move to GPU if available
    x_mem = sp.GetMemoryAndReshape(node.Input(IN_X), x_mem.get_desc(),
                                   zendnn_engine);
    x_scale_mem = sp.GetMemoryAndReshape(node.Input(IN_X_SCALE),
                                         x_scale_mem.get_desc(), zendnn_engine);
    // Get descs
    auto x_md = x_mem.get_desc();
    auto x_scale_md = x_scale_mem.get_desc();
    auto x_dims = x_md.dims().size();

    // Fix scale dims
    int64_t axis = GetAxis(node, x_dims);
    // Check if axis is negative and fix it
    if (axis < 0) {
        axis += x_dims;
    }
    // Prepare the scale to prevent broacasting errors
    if (isScalar) {
        // For scalar scale
        Padd(&x_scale_md, x_dims, false);
    }
    else {
        // For N-D scale
        Padd(&x_scale_md, static_cast<uint64_t>(axis) + 1, x_dims);
    }

    // Create dst mem
    auto dst_md = zendnn::memory::desc(x_md.dims(), node.Output(OUT_Y).Type(),
                                       zendnn::memory::format_tag::any);
    zendnn::memory dst_mem;

    // If zero point exists and we are NOT dequantizing int32, then substract zp from x and scale
    if (isZeroPointUseful &&
            (x_mem.get_desc().data_type() != zendnn::memory::data_type::s32)) {
        // Get Zero point
        auto x_zp_mem = sp.GetMemory(node.Input(IN_X_ZERO_POINT));
        // Get mds for operands
        auto x_zp_md = x_zp_mem.get_desc();

        // Prepare the zp to prevent broacasting errors
        if (isScalar) {
            // For scalar zp
            Padd(&x_zp_md, x_dims, false);
        }
        else {
            // For N-D zp
            Padd(&x_zp_md, static_cast<uint64_t>(axis) + 1, x_dims);
        }

        // Create binary desc
        auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_sub, x_md,
                                             x_zp_md, dst_md);
        // Add post op scale
        zendnn::primitive_attr binary_attr;
        {
            zendnn::post_ops binary_ops;
            binary_ops.append_binary(zendnn::algorithm::binary_mul, x_scale_md);
            binary_attr.set_post_ops(binary_ops);
        }
        // Add post op to scale result
        auto binary_pd = zendnn::binary::primitive_desc(binary_d, binary_attr,
                         zendnn_engine);
        // Move to GPU if available
        x_zp_mem = sp.GetMemoryAndReshape(node.Input(IN_X_ZERO_POINT), x_zp_md,
                                          zendnn_engine);
        // Create primitive and set dst mem
        dst_mem = zendnn::memory(binary_pd.dst_desc(), zendnn_engine);
        auto binary_prim = zendnn::binary(binary_pd);

        sp.AddPrimitive(binary_prim, {{ZENDNN_ARG_SRC_0, x_mem},
            {ZENDNN_ARG_SRC_1, x_zp_mem},
            {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, x_scale_mem},
            {ZENDNN_ARG_DST, dst_mem}
        });

        // If zp doesn't exists or we are dequantizing from int32, only need to scale
    }
    else {
        // Create binary and primitive desc
        auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_mul, x_md,
                                             x_scale_md, dst_md);
        auto binary_pd = zendnn::binary::primitive_desc(binary_d, zendnn_engine);

        // Create primitive
        dst_mem = zendnn::memory(binary_pd.dst_desc(), zendnn_engine);
        auto binary_prim = zendnn::binary(binary_pd);

        sp.AddPrimitive(binary_prim, {{ZENDNN_ARG_SRC_0, x_mem},
            {ZENDNN_ARG_SRC_1, x_scale_mem},
            {ZENDNN_ARG_DST, dst_mem}
        });
    }

    // Set the output mem
    if (sp.IsScalar(node.Input(IN_X))) {
        sp.SetMemory(node.Output(OUT_Y), dst_mem, false, true);
    }
    else {
        sp.SetMemory(node.Output(OUT_Y), dst_mem);
    }
}

bool ZendnnDequantizeLinear::isZeroPointNonZero(zendnn::memory *zp_mem) {
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

void ZendnnDequantizeLinear::Padd(zendnn::memory::desc *target_md,
                                  size_t front_pad, size_t back_pad) {
    // Pads an input to broadcast the op correctly
    auto target_dims = target_md->dims();

    // Add front padding
    while (target_dims.size() < front_pad) {
        target_dims.insert(target_dims.begin(), 1);
    }
    // Add back padd
    while (target_dims.size() < back_pad) {
        target_dims.insert(target_dims.end(), 1);
    }

    *target_md = target_md->reshape(target_dims);
}

int64_t ZendnnDequantizeLinear::GetAxis(ZendnnNode &node, size_t x_dims) {
    // We need to do sign comparisons so we have to cast
    int64_t sig_x_dims = static_cast<uint64_t>(x_dims);
    auto attr = node.Attributes().find("axis");
    // If axis is provided, make sure axis is an integer and
    // has a range of [-r, r]
    if (attr != node.Attributes().end()) {
        int64_t axis2 = attr->second().i();
        if (attr->second().type() ==
                ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT
                &&
                (((axis2 <= 0) && (axis2 >= -sig_x_dims)) ||
                 ((axis2 >= 0) && (axis2 <= (sig_x_dims - 1))))) {
            return attr->second().i();
        }
    }
    // Return the default value
    return 1;
}

void ZendnnDequantizeLinear::ValidateDims(ZendnnSubgraphPrimitive &sp,
        ZendnnNode &node) {
    // We only need to validate when zp is provided
    if (node.Input(IN_X_ZERO_POINT).Exists()) {
        auto x_scale_dims = sp.GetMemory(node.Input(IN_X_SCALE)).get_desc().dims();
        auto x_zp_dims = sp.GetMemory(node.Input(IN_X_ZERO_POINT)).get_desc().dims();

        if (x_zp_dims != x_scale_dims) {
            ORT_THROW("x_scale and x_zero_point dimensions does not match");
        }
    }
}

void ZendnnDequantizeLinear::ValidateType(ZendnnSubgraphPrimitive &sp,
        ZendnnNode &node) {
    // If zp exists check its dataype
    if (node.Input(IN_X_ZERO_POINT).Exists()) {
        auto x_md = sp.GetMemory(node.Input(IN_X)).get_desc();
        auto x_zp_md = sp.GetMemory(node.Input(IN_X_ZERO_POINT)).get_desc();

        if (x_md.data_type() != x_zp_md.data_type()) {
            ORT_THROW("x and x_zero_point have different datatypes");
        }
    }
}

}  // namespace ort_zendnn
}  // namespace onnxruntime