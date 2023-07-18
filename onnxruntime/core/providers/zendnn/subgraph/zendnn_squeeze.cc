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

#include "zendnn_squeeze.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace ort_zendnn {
ZendnnSqueeze::ZendnnSqueeze() {}

void ZendnnSqueeze::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                    ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    // the input shape assumes OrtFormat so we get the memory in OrtFormat.
    auto data_mem = sp.GetMemoryInOrtFormat(node.Input(IN_DATA), zendnn_engine);
    zendnn::memory::dims data_dims = data_mem.get_desc().dims();

    std::vector<int64_t> axes_data;
    // ONNX Squeeze version 13+ the axes is an input tensor
    // ONNX Squeeze before version 13 axes comes from an Attribute.
    if (node.Input(IN_AXES).Exists()) {
        auto axes_mem = sp.GetMemory(node.Input(IN_AXES));
        zendnn::memory::dims axes_dims = axes_mem.get_desc().dims();
        int64_t *p_axes_data = (int64_t *)axes_mem.get_data_handle();
        axes_data = std::vector<int64_t>(p_axes_data, p_axes_data + axes_dims[0]);
    }
    else {
        axes_data = GetAxes(node);
    }

    // convert negative axis to the positive axis
    for (size_t i = 0; i < axes_data.size(); ++i) {
        axes_data[i] = HandleNegativeAxis(axes_data[i], data_dims.size());
    }

    // Handle out of order and repeating dims.
    std::sort(axes_data.begin(), axes_data.end());
    axes_data.erase(std::unique(axes_data.begin(), axes_data.end()),
                    axes_data.end());

    std::vector<int64_t> output_shape;
    size_t j = 0;
    for (size_t i = 0; i < data_dims.size(); ++i) {
        if ((j < axes_data.size() && axes_data[j] == static_cast<int64_t>(i)) ||
                (axes_data.size() == 0 && data_dims[i] == 1)) {
            ORT_ENFORCE(data_dims[i] == 1, "Dimension of input ", i,
                        " must be 1 instead of ", data_dims[i],
                        ". shape=", TensorShape(data_dims));
            ++j;
            continue;
        }
        output_shape.push_back(data_dims[i]);
    }

    zendnn::memory::desc squeeze_md(output_shape, node.Input(IN_DATA).Type(),
                                    sp.GetZendnnFormat(output_shape.size()));

    zendnn::memory squeeze_mem = zendnn::memory(squeeze_md, zendnn_engine, nullptr);
    sp.AddReshape(data_mem, squeeze_mem);

    sp.SetMemory(node.Output(OUT_SQUEEZED), squeeze_mem, true);
}

std::vector<int64_t> ZendnnSqueeze::GetAxes(ZendnnNode &node) {
    auto attr = node.Attributes().find("axes");
    std::vector<int64_t> axes;
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS) {
        axes.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            axes.push_back(attr->second().ints(i));
        }
    }
    return axes;
}
}  // namespace ort_zendnn
}  // namespace onnxruntime
