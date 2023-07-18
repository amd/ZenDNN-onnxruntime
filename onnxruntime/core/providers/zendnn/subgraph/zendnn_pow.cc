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

#include "zendnn_pow.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnPow::ZendnnPow() {}

void ZendnnPow::CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    auto elementwise_src_mem = sp.GetMemory(node.Input(IN_X));
    auto src_md = elementwise_src_mem.get_desc();

    auto exponent_src_mem = sp.GetMemory(node.Input(IN_Y));

    float beta = 1.0;
    switch (node.Input(IN_Y).Type()) {
    case zendnn::memory::data_type::f32: {
        beta = static_cast<float>(*(float *)exponent_src_mem.get_data_handle());
        break;
    }
    case zendnn::memory::data_type::s32: {
        beta = static_cast<float>(*(int32_t *)exponent_src_mem.get_data_handle());
        break;
    }
    case zendnn::memory::data_type::s8: {
        beta = static_cast<float>(*(int8_t *)exponent_src_mem.get_data_handle());
        break;
    }
    case zendnn::memory::data_type::u8: {
        beta = static_cast<float>(*(uint8_t *)exponent_src_mem.get_data_handle());
        break;
    }
    case zendnn::memory::data_type::bf16: {
        beta = static_cast<float>(*(BFloat16 *)exponent_src_mem.get_data_handle());
        break;
    }
    default:
        ORT_THROW("Pow exponent data type not supported");
    }

    // ZENDNN eltwise_pow is defined as alpha*x^beta. We don't use alpha so it is hard coded to 1.0
    zendnn::eltwise_forward::desc elementwise_desc(
        zendnn::prop_kind::forward_inference, zendnn::algorithm::eltwise_pow, src_md,
        1.0, beta);
    zendnn::eltwise_forward::primitive_desc elementwise_pd(elementwise_desc,
            zendnn_engine);

    // If using GPU this will move the memory from the CPU to the GPU.
    elementwise_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X),
                          elementwise_pd.src_desc(), zendnn_engine);
    auto elementwise_dst_mem = zendnn::memory(elementwise_pd.dst_desc(),
                               zendnn_engine);

    auto elemenwise_primitive = zendnn::eltwise_forward(elementwise_pd);
    sp.AddPrimitive(elemenwise_primitive, {{ZENDNN_ARG_SRC, elementwise_src_mem},
        {ZENDNN_ARG_DST, elementwise_dst_mem}
    });

    sp.SetMemory(node.Output(OUT_Z), elementwise_dst_mem);
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
