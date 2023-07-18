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

#include "zendnn_unsqueeze.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace ort_zendnn {
ZendnnUnsqueeze::ZendnnUnsqueeze() {}

void ZendnnUnsqueeze::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                      ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    // the input shape assumes OrtFormat so we get the memory in OrtFormat.
    auto data_mem = sp.GetMemoryInOrtFormat(node.Input(IN_DATA), zendnn_engine);
    bool data_is_scalar = sp.IsScalar(node.Input(IN_DATA));

    // The ZenDNN execution provider automatically expands all scalar inputs to dim {1} tensors.
    // this will result in the data_dims.size() being 1 too large if the input is from a scalar.
    // To counter this data_dims is left empty if the input is from a scalar.
    zendnn::memory::dims data_dims;
    if (!data_is_scalar) {
        data_dims = data_mem.get_desc().dims();
    }

    std::vector<int64_t> axes_data;
    // ONNX Unsqueeze version 13+ the axes is an input tensor
    // ONNX Unsqueeze before version 13 axes comes from an Attribute.
    if (node.Input(IN_AXES).Exists()) {
        auto axes_mem = sp.GetMemory(node.Input(IN_AXES));
        zendnn::memory::dims axes_dims = axes_mem.get_desc().dims();
        int64_t *p_axes_data = (int64_t *)axes_mem.get_data_handle();
        axes_data = std::vector<int64_t>(p_axes_data, p_axes_data + axes_dims[0]);
    }
    else {
        axes_data = GetAxes(node);
    }

    std::vector<int64_t> output_shape(axes_data.size() + data_dims.size(), 0);
    // Set all axes indices to 1 in output_dims and check for duplicates
    for (int64_t axes : axes_data) {
        // Valid axis range is [0, output_rank - 1]
        axes = HandleNegativeAxis(axes, output_shape.size());
        if (axes < 0 || axes >= static_cast<int64_t>(output_shape.size())) {
            ORT_ENFORCE("'axes' has an out of range axis");
        }
        if (output_shape[axes] != 0) {
            ORT_ENFORCE("'axes' has a duplicate axis");
        }
        output_shape[axes] = 1;
    }

    // Now fill in the zero entries with the existing shape
    {
        auto begin = data_dims.cbegin();
        for (auto &axisSize : output_shape) {
            if (axisSize == 0) {
                axisSize = *begin++;
            }
        }
        assert(begin == data_dims.cend());
    }

    zendnn::memory::desc squeeze_md(output_shape, node.Input(IN_DATA).Type(),
                                    sp.GetZendnnFormat(output_shape.size()));

    zendnn::memory expanded_mem = zendnn::memory(squeeze_md, zendnn_engine,
                                  nullptr);
    sp.AddReshape(data_mem, expanded_mem);

    sp.SetMemory(node.Output(OUT_EXPANDED), expanded_mem, true);
}

std::vector<int64_t> ZendnnUnsqueeze::GetAxes(ZendnnNode &node) {
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
    else {
        ORT_ENFORCE("Missing/Invalid 'axes' attribute value");
    }
    return axes;
}
}  // namespace ort_zendnn
}  // namespace onnxruntime
