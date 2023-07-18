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

#include "zendnn_flatten.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnFlatten::ZendnnFlatten() {}

void ZendnnFlatten::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                    ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    auto data_mem = sp.GetMemory(node.Input(IN_DATA));
    zendnn::memory::dims data_dims = data_mem.get_desc().dims();

    // Get the axis
    int64_t axis = GetAxis(node);
    if (axis < 0) {
        axis = data_dims.size() + axis;
    }

    std::vector<int64_t> dst_dims;
    dst_dims.reserve(2);

    dst_dims[0] = 1;
    dst_dims[1] = 1;

    if (axis != 0) {
        for (int64_t i = 0; i < axis; i++) {
            dst_dims[0] *= data_dims[i];
        }
    }

    for (uint64_t i = axis; i < (uint64_t)data_dims.size(); i++) {
        dst_dims[1] *= data_dims[i];
    }

    zendnn::memory::dims dst_dims_zendnn(dst_dims.begin(), dst_dims.end());
    //zendnn::memory::format_tag dst_format = zendnn::memory::format_tag::nc;
    zendnn::memory::dims dst_strides = zendnn::memory::dims{dst_dims[1], zendnn::memory::dim(1)};

    zendnn::memory::desc dst_md({dst_dims[0],dst_dims[1]}, node.Input(
                                    IN_DATA).Type(), dst_strides);
    data_mem = sp.GetMemoryInOrtFormat(node.Input(IN_DATA), zendnn_engine);

    auto dst_mem = zendnn::memory(dst_md, zendnn_engine, nullptr);
    sp.AddReshape(data_mem, dst_mem);
    sp.SetMemory(node.Output(OUT_FLATTEN), dst_mem, true);
}

int64_t ZendnnFlatten::GetAxis(ZendnnNode &node) {
    auto attr = node.Attributes().find("axis");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return 1;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
