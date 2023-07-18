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

#include "zendnn_cast.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnCast::ZendnnCast() {}

void ZendnnCast::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                 ZendnnNode &node) {

    // Get the ZENDNN engine
    auto zendnn_engine = sp.GetEngine();

    // Get the memory from the input node
    auto src_mem = sp.GetMemory(node.Input(IN_INPUT));
    auto src_tag = node.Input(IN_INPUT).Format();
    auto src_md = src_mem.get_desc();
    auto src_dims = src_md.dims();

    // dst characteristics
    zendnn::memory::data_type dst_type;
    zendnn::memory::format_tag dst_tag;

    // Get the target data type
    auto dst_type_desc = GetTo(node);

    // Check fot the target datat ype
    switch (dst_type_desc) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        dst_type = zendnn::memory::data_type::f32;
        break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        dst_type = zendnn::memory::data_type::f16;
        break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
        dst_type = zendnn::memory::data_type::bf16;
        break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        dst_type = zendnn::memory::data_type::s32;
        break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
        dst_type = zendnn::memory::data_type::s8;
        break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
        dst_type = zendnn::memory::data_type::u8;
        break;
    }
    default:
        ORT_THROW("Unsupported data type: ", dst_type_desc);
        break;
    }
    // Be aware that the output memory will be in plain format
    // and depending on the operation you do next, this wont be as
    // efficient as you'd like
    // If the format tag is any
    if (src_tag == zendnn::memory::format_tag::any) {
        // Define a plain data ND format
        dst_tag = sp.GetZendnnFormat(src_dims.size());
    }
    else {
        // Else use the same as the source
        dst_tag = src_tag;
    }

    // Generate the dst memory descriptor
    auto dst_md = zendnn::memory::desc(src_md.dims(), dst_type, dst_tag);

    // Create the reorder primitive descriptor.
    auto reorder_pd = zendnn::reorder::primitive_desc(zendnn_engine, src_md,
                      zendnn_engine, dst_md);
    // Get the dst memory
    auto dst_mem = zendnn::memory(reorder_pd.dst_desc(), zendnn_engine);

    // If using GPU this will move the memory from the CPU to the GPU.
    src_mem = sp.GetMemoryAndReshape(node.Input(IN_INPUT), reorder_pd.src_desc(),
                                     zendnn_engine);

    // ZenDNN uses reorder to cast the src_md data to the dst_md data type
    auto reorder = zendnn::reorder(reorder_pd);

    // Add primitive to the graph
    sp.AddPrimitive(reorder, {{ZENDNN_ARG_SRC, src_mem},
        {ZENDNN_ARG_DST, dst_mem}
    });

    // Support scalar return values
    if (sp.IsScalar(node.Input(OUT_OUTPUT))) {
        sp.SetMemory(node.Output(OUT_OUTPUT), dst_mem, false, true);
    }
    else {
        sp.SetMemory(node.Output(OUT_OUTPUT), dst_mem);
    }

}

int64_t ZendnnCast::GetTo(ZendnnNode &node) {
    // Get the attribute
    auto attr = node.Attributes().find("to");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    else {
        // to attribute should always exist in order to cast
        ORT_THROW("TO(CAST TARGET DATA TYPE) DOES NOT EXIST");
    }
}

}  // namespace ort_zendnn
}  // namespace onnxruntime